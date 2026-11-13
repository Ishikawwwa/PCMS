"""Loopback-only browser console for the Polza concept-memory experiments."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import subprocess
import sys
from threading import RLock
from urllib.parse import parse_qs, urlsplit
import webbrowser

from constraint_memory.secrets import load_env, redact

from .lab_trace import atomic_json, read_json
from .run_live import ROOT


UI = Path(__file__).with_name("lab_ui")
RUN_ID = re.compile(r"concept-subspaces-[a-z0-9-]+\Z")
SHOPPING = ROOT / "tmp/concept-subspaces-shopping-release.jsonl"
TRAVEL = ROOT / "data/hidden/memoryarena_travel.jsonl"
TRAIN = ROOT / "tmp/concept-subspaces-travel-train.csv"
TEXT_SUFFIXES = {".log", ".json", ".jsonl", ".txt"}
FILE_PAGE = 256 * 1024
TERMINAL = {"complete", "failed", "stopped", "terminated"}
BREAKPOINTS = {"model", "memory", "retrieval", "tool", "score", "arm"}


class Conflict(ValueError):
    pass


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


class Runs:
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.controls = self.directory / ".concept-lab"
        self.controls.mkdir(exist_ok=True)
        self.children = {}
        self.forced = set()
        self.lock = RLock()

    def folder(self, run):
        if not isinstance(run, str) or not RUN_ID.fullmatch(run):
            raise FileNotFoundError("Unknown run")
        candidate = (self.directory / run).resolve()
        if candidate.parent != self.directory or (not candidate.is_dir() and run not in self.children):
            raise FileNotFoundError("Unknown run")
        return candidate

    def control_path(self, run):
        return self.controls / f"{run}.json"

    def launch(self, options):
        if not isinstance(options, dict):
            raise ValueError("Expected run options")
        experiment = options.get("experiment")
        memory_tokens = integer(options.get("memory_tokens"), "Memory tokens", 1, 4096)
        if experiment in {"shopping", "travel"}:
            module = "run_live"
            args = ["--case", experiment]
            if experiment == "shopping":
                if not SHOPPING.is_file():
                    raise ValueError("The pinned shopping release file is missing. See the setup instructions.")
                args += ["--shopping-release", str(SHOPPING)]
        elif experiment in {"paired", "benchmark"}:
            max_steps = integer(options.get("max_steps"), "Action limit", 1, 100)
            thinking = options.get("thinking")
            if type(thinking) is not bool:
                raise ValueError("Choose an actor reasoning setting")
            if not (TRAVEL.is_file() and TRAIN.is_file() and (ROOT / "external/MAGE/benchmark/travelplan").is_dir()):
                raise ValueError("The pinned MAGE source or travel inputs are missing. See the setup instructions.")
            args = ["--max-steps", str(max_steps), "--travel-release", str(TRAVEL), "--train-csv", str(TRAIN)]
            if experiment == "benchmark":
                module = "run_travel_benchmark"
                args += ["--workers", str(integer(options.get("workers", 8), "Workers", 1, 16))]
                if options.get("pcms_only") is True:
                    args.append("--pcms-only")
                if options.get("verify_repair_first") is True:
                    args.append("--verify-repair-first")
                if thinking:
                    args.append("--actor-thinking")
            else:
                module = "run_mage_travel"
                group = integer(options.get("group"), "Development group", 1, 270)
                if group not in self.groups():
                    raise ValueError("Select a development group")
                args += ["--group", str(group), "--actor-thinking" if thinking else "--no-actor-thinking"]
                if options.get("turns") is not None:
                    args += ["--turns", str(integer(options["turns"], "Traveler limit", 1, 20))]
        else:
            raise ValueError("Choose an experiment")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run = f"concept-subspaces-lab-{experiment}-{stamp}-{secrets.token_hex(3)}"
        out = self.directory / run
        control = self.control_path(run)
        command = [sys.executable, "-X", "utf8", "-m", f"experiments.concept_subspaces.{module}", *args,
                   "--memory-tokens", str(memory_tokens), "--out", str(out), "--lab-control", str(control)]
        with self.lock:
            atomic_json(control, {"mode": "pause", "revision": 0})
            with (self.controls / f"{run}.log").open("wb") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.children[run] = process
        return run

    def groups(self):
        manifest = read_json(ROOT / "data/manifests/splits.json", {})
        return manifest.get("partitions", {}).get("dev", [])

    def describe(self, run):
        folder = self.folder(run)
        manifest_path = (folder / "manifest.json").resolve()
        if manifest_path.parent != folder:
            raise FileNotFoundError("Run manifest is outside the run")
        manifest = read_json(manifest_path, {})
        with self.lock:
            process = self.children.get(run)
            result = {"id": run, "manifest": manifest, "timeline": (folder / "timeline.jsonl").exists(),
                      "managed": process is not None, "status": manifest.get("status", "archived")}
            if process is None:
                return result
            exit_code = process.poll()
            result["exit_code"] = exit_code
            if exit_code is not None:
                if result["status"] not in TERMINAL:
                    result["status"] = "terminated" if run in self.forced else "failed"
                return result
            command = read_json(self.control_path(run))
            state = read_json(self.control_path(run).with_suffix(".state.json"), {})
            result.update(command=command, checkpoint=state)
            if result["status"] in TERMINAL:
                return result
            if command["mode"] in {"stop", "terminate"}:
                result["status"] = "stop_requested"
            elif not state:
                result["status"] = "starting"
            elif state["revision"] != command["revision"]:
                result["status"] = {"pause": "pause_requested", "run": "resuming", "step": "executing_step",
                                    "run_to": "resuming"}[command["mode"]]
            else:
                result["status"] = state["status"]
            return result

    def list(self):
        with self.lock:
            names = set(self.children)
            names.update(path.name for path in self.directory.iterdir() if path.is_dir() and RUN_ID.fullmatch(path.name))
        items = []
        for name in names:
            try:
                item = self.describe(name)
                folder = self.folder(name)
                marker = folder / "manifest.json"
                updated = marker.stat().st_mtime if marker.is_file() else self.control_path(name).stat().st_mtime
                items.append({**item, "updated": updated})
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                continue
        return sorted(items, key=lambda item: (item["updated"], item["id"]), reverse=True)

    def command(self, run, action, target=None):
        with self.lock:
            info = self.describe(run)
            process = self.children.get(run)
            if process is None or process.poll() is not None or info["status"] in TERMINAL:
                raise Conflict("This run is read-only")
            if action == "terminate":
                if info["manifest"].get("benchmark") or "-lab-benchmark-" in run:
                    atomic_json(self.control_path(run), {"mode": "terminate", "revision": info["command"]["revision"] + 1})
                    return self.describe(run)
                self.forced.add(run)
                process.terminate()
                return self.describe(run)
            if action not in {"run", "run_to", "pause", "step", "stop"}:
                raise ValueError("Unknown control command")
            if info["manifest"].get("benchmark") and action in {"step", "run_to"}:
                raise ValueError("Parallel benchmarks support pause, resume and stop")
            if info["status"] == "stop_requested":
                raise Conflict("The run is stopping")
            if action == "step" and info["status"] != "paused":
                raise Conflict("Wait until the worker reaches a paused checkpoint before stepping")
            control = self.control_path(run)
            command = {"mode": action, "revision": info["command"]["revision"] + 1}
            if action == "run_to":
                if target not in BREAKPOINTS:
                    raise ValueError("Choose a breakpoint")
                command["target"] = target
                if info["status"] == "paused":
                    command["skip"] = info["checkpoint"]["checkpoint"]
            atomic_json(control, command)
            return self.describe(run)

    def files(self, run):
        folder = self.folder(run)
        files = []
        for path in folder.rglob("*"):
            if path.suffix in TEXT_SUFFIXES and not path.name.startswith(".") and path.resolve().is_relative_to(folder):
                try:
                    if path.is_file():
                        files.append({"path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size})
                except FileNotFoundError:
                    continue
        log = self.controls / f"{run}.log"
        if log.is_file():
            files.append({"path": "@worker.log", "bytes": log.stat().st_size})
        return sorted(files, key=lambda entry: entry["path"])

    def file(self, run, name):
        folder = self.folder(run)
        if name == "@worker.log":
            path = (self.controls / f"{run}.log").resolve()
            if path.parent == self.controls and path.is_file():
                return path
            raise FileNotFoundError("Artifact not found")
        relative = Path(name)
        path = (folder / relative).resolve()
        if (relative.is_absolute() or any(part.startswith(".") or ":" in part for part in relative.parts)
                or not path.is_relative_to(folder) or path.suffix not in TEXT_SUFFIXES or not path.is_file()):
            raise FileNotFoundError("Artifact not found")
        return path


def event_batch(path, offset):
    events = []
    if not path.exists():
        return {"next": 0, "events": events}
    with path.open("rb") as stream:
        stream.seek(min(offset, path.stat().st_size))
        while len(events) < 200:
            start = stream.tell()
            line = stream.readline()
            if not line.endswith(b"\n"):
                return {"next": start, "events": events}
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            data = row.get("data", {})
            scope = data.get("scope", {})
            summary = {key: row[key] for key in ("seq", "at", "kind", "title")}
            summary.update(offset=start, scope=scope, condition=data.get("condition"), turn=data.get("turn"),
                           mode=data.get("mode"), evidence_id=data.get("evidence_id"),
                           tokens=data.get("total_tokens"), cost=data.get("cost_rub"))
            events.append(summary)
        return {"next": stream.tell(), "events": events}


def file_page(path, end):
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        end = size if end is None else min(end, size)
        start = max(0, end - FILE_PAGE)
        stream.seek(start)
        return {"start": start, "end": end, "size": size,
                "text": redact(stream.read(end - start).decode("utf-8", errors="replace"))}


class LabHTTP(ThreadingHTTPServer):
    def __init__(self, address, runs):
        super().__init__(address, Handler)
        self.runs = runs
        self.token = secrets.token_urlsafe(24)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)

    def json(self, status, data):
        self.send(status, json.dumps(redact(data), ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def dispatch(self, write):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            self.json(403, {"error": "The console only accepts local requests"})
            return
        target = urlsplit(self.path)
        query = {key: values[-1] for key, values in parse_qs(target.query).items()}
        runs = self.server.runs
        try:
            if write:
                if (self.headers.get("X-Lab-Control") != self.server.token
                        or self.headers.get("Origin") not in {f"http://{host}" for host in hosts}):
                    self.json(403, {"error": "Invalid local control session"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid control request length")
                data = json.loads(self.rfile.read(length))
                if target.path == "/api/start":
                    self.json(201, {"run": runs.launch(data)})
                elif target.path == "/api/control" and isinstance(data, dict):
                    self.json(200, runs.command(data.get("run"), data.get("action"), data.get("target")))
                else:
                    raise FileNotFoundError("Unknown endpoint")
                return

            static = {"/": ("index.html", "text/html"), "/lab.css": ("lab.css", "text/css"),
                      "/lab.js": ("lab.js", "text/javascript")}
            if target.path in static:
                name, kind = static[target.path]
                self.send(200, (UI / name).read_bytes(), kind + "; charset=utf-8")
                return
            if target.path == "/api/config":
                self.json(200, {"token": self.server.token, "groups": runs.groups(), "shopping_ready": SHOPPING.is_file(),
                                "paired_ready": TRAVEL.is_file() and TRAIN.is_file()})
                return
            if target.path == "/api/runs":
                self.json(200, runs.list())
                return
            run = query.get("run")
            folder = runs.folder(run)
            if target.path == "/api/events":
                path = folder / "timeline.jsonl"
                if path.resolve().parent != folder:
                    raise FileNotFoundError("Timeline is outside the run")
                self.json(200, event_batch(path, max(0, int(query.get("after", "0")))))
            elif target.path == "/api/benchmark":
                self.json(200, read_json(runs.file(run, "benchmark.json")))
            elif target.path == "/api/event":
                with runs.file(run, "timeline.jsonl").open("rb") as stream:
                    stream.seek(max(0, int(query["offset"])))
                    self.json(200, json.loads(stream.readline()))
            elif target.path == "/api/files":
                self.json(200, runs.files(run))
            elif target.path == "/api/file":
                end = max(0, int(query["end"])) if "end" in query else None
                self.json(200, file_page(runs.file(run, query.get("path", "")), end))
            else:
                raise FileNotFoundError("Unknown endpoint")
        except FileNotFoundError as error:
            self.json(404, {"error": str(error)})
        except Conflict as error:
            self.json(409, {"error": str(error)})
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.json(400, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            return
        except OSError as error:
            self.json(500, {"error": str(error)})

    def do_GET(self):
        self.dispatch(write=False)

    def do_POST(self):
        self.dispatch(write=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the console in the default browser")
    args = parser.parse_args()
    load_env(ROOT / ".env")
    with LabHTTP(("127.0.0.1", args.port), Runs(ROOT / "runs")) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Concept Memory Lab is listening at {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            for run, child in list(server.runs.children.items()):
                if child.poll() is None:
                    try:
                        server.runs.command(run, "stop")
                    except Conflict:
                        pass


if __name__ == "__main__":
    main()
