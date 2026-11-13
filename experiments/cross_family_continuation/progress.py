"""Console-only observations. No prompts, responses, tools or budgets change."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
import json
import math
from pathlib import Path
import threading
import time

from experiments.cross_family_continuation import core
from constraint_memory.secrets import redact

MSK = dt.timezone(dt.timedelta(hours=3), "MSK")


def safe(value, limit=100):
    return str(redact(str(value))).replace("\n", " ").replace("\r", " ")[:limit]


def duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def observe(callback, *args, **kwargs):
    """Optional console callbacks cannot change a request or its exception."""
    try:
        return callback(*args, **kwargs)
    except Exception:
        return None


def observed_clock(progress):
    value = observe(progress.clock)
    return value if type(value) in (int, float) and math.isfinite(value) else time.monotonic()


class ProgressReporter:
    def __init__(self, parent=None, *, sink=print, clock=time.monotonic, heartbeat_seconds=10, background=True):
        self.sink = (lambda message: print(message, flush=True)) if sink is print else sink
        self.clock = clock
        self.started = self.phase_started = clock()
        self.phase_name = "Starting"
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.heartbeat_seconds = heartbeat_seconds
        self.parent = None
        self.completed = set()
        self.closed_sessions = set()
        self.total_sessions, self.total_arrivals = 14, 98
        self.history = []
        self.new_results, self.new_posts, self.new_known = 0, 0, Decimal(0)
        self.unknown = 0
        self.known_before = Decimal(0)
        self.current_scope, self.arrival, self.traveler = None, None, None
        self.actor_used = 0
        self.arrival_model_seconds = 0
        self.stream_directory = None
        self.thread = None
        if parent is not None:
            self.configure(parent)
        if background:
            self.thread = threading.Thread(target=self._loop, name="benchmark-console-progress", daemon=True)
            self.thread.start()

    def _line(self, message):
        with self.lock:
            timestamp = dt.datetime.now(MSK).strftime("%H:%M:%S MSK")
            self.sink(f"[{timestamp} +{duration(self.clock() - self.started)}] {message}")

    def _loop(self):
        while not self.stop.wait(self.heartbeat_seconds):
            try:
                self.heartbeat()
            except Exception:
                # Console observation must never interrupt an in-flight POST.
                pass

    def configure(self, parent):
        self.parent = parent
        schedule = parent["protocol"]["schedule"]
        self.total_sessions = len(schedule)
        self.total_arrivals = parent["protocol"].get("bounds", {}).get("added_traveler_commitments", len(schedule) * 7)
        self.closed_sessions = {(item["variant"], str(item["group_id"])) for item in parent["completed"]}
        traces = core.base.rows(parent["directory"] / "traces.jsonl")
        self.completed = {(f"{row['variant']}:{row['group_id']}", int(row["charged_round"])) for row in traces}
        self.known_before = sum((Decimal(str(entry["usage"]["cost_rub"])) for entry in parent["accepted"]
                                if entry["usage"].get("cost_rub") is not None), Decimal(0))
        self.unknown = sum(row.get("cost_rub") is None for row in parent.get("all_failures", []))
        counters, times = {}, {}
        for entry in parent["accepted"]:
            intent = entry["intent"]
            scope = intent["group_id"]
            if intent["operation"] == "extraction":
                counters[scope] = counters.get(scope, 0) + 1
            key = (scope, counters.get(scope, 0))
            latency = entry["usage"].get("latency_seconds")
            if key in self.completed and type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
                times[key] = times.get(key, 0) + latency
        self.history = [value for value in times.values() if value > 0]
        self._line(f"SAVED WORK: {len(parent['accepted'])} accepted responses; {self.summary()}")
        self._line("Progress counts processed travelers, including recorded unsuccessful outcomes. ETA is approximate model time; scoring follows planning.")

    def summary(self):
        if self.parent is None:
            return "saved progress is loading"
        remaining = max(0, self.total_arrivals - len(self.completed))
        summary = (f"sessions {len(self.closed_sessions)}/{self.total_sessions} closed; "
                   f"travelers {len(self.completed)}/{self.total_arrivals} processed ({remaining} left)")
        if remaining and len(self.history) >= 4:
            samples = sorted(self.history)
            low, high = samples[len(samples) // 4], samples[(3 * len(samples)) // 4]
            summary += f"; rough model-time ETA {duration(low * remaining)}--{duration(high * remaining)}"
        elif not remaining:
            summary += "; planning complete"
        return summary

    def phase(self, message):
        with self.lock:
            full_message = safe(message, 700)
            self.phase_name = full_message.split("; sessions ", 1)[0]
            self.phase_started = self.clock()
            self._line(full_message)

    def session(self, index, variant, group):
        self.current_scope = f"{variant}:{group}"
        self.phase(f"SESSION {index}/{self.total_sessions} {safe(self.current_scope)}; {self.summary()}")

    def session_done(self, variant, group):
        with self.lock:
            self.closed_sessions.add((variant, str(group)))
            self.phase(f"SESSION COMPLETE {safe(variant)}:{safe(group)}; {self.summary()}")

    def arrival_start(self, scope, index, traveler, replay=False):
        self.current_scope, self.arrival, self.traveler = scope, index, traveler
        self.arrival_model_seconds = 0
        debits = (self.parent or {}).get("failure_debits", {}).get(scope, {})
        self.actor_used = debits.get(index, debits.get(str(index), 0))
        self.phase(f"{'REPLAY' if replay else 'LIVE'} TRAVELER {index}/7 {safe(traveler)} in {safe(scope)}; "
                   f"{30 - self.actor_used} actor calls available; {self.summary()}")

    def arrival_done(self, scope, index, traveler, elapsed, replayed=False):
        with self.lock:
            key = (scope, index)
            was_completed = key in self.completed
            self.completed.add(key)
            if not replayed and not was_completed and self.arrival_model_seconds > 0:
                self.history.append(self.arrival_model_seconds)
            self.phase(f"TRAVELER PROCESSED {index}/7 {safe(traveler)} in {duration(elapsed)}"
                       f"{' (saved work replayed)' if replayed else ''}; {self.summary()}")

    def request_start(self, operation, scope, traveler, cached=False):
        if cached:
            return
        label = "MEMORY WRITER" if operation == "extraction" else f"ACTOR call {self.actor_used + 1}/30"
        self.phase(f"{label}: {safe(scope)} / {safe(traveler)}; preparing next request")

    def request_result(self, record, *, cached=False, finish_reason=None):
        operation = record.get("operation")
        latency = record.get("latency_seconds")
        if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
            self.arrival_model_seconds += latency
        if operation == "agent" and (record.get("request_status") == "completed" or record.get("request_sent") is True):
            self.actor_used += 1
        if cached:
            return
        accepted = record.get("request_status") == "completed" and record.get("response_accepted") is True
        sent = accepted or record.get("request_sent") is True
        self.new_posts += int(sent)
        self.new_results += int(accepted)
        cost = record.get("cost_rub")
        if cost is not None:
            self.new_known += Decimal(str(cost))
        else:
            self.unknown += 1
        if accepted:
            self.phase(f"RESPONSE ACCEPTED: {safe(operation)}; finish={safe(finish_reason)}; "
                       f"tokens in/out {record.get('input_tokens')}/{record.get('output_tokens')}; "
                       f"cost {cost} RUB; elapsed {duration(record.get('latency_seconds', 0))}; "
                       f"new accepted {self.new_results}, sent {self.new_posts}; known new spend {self.new_known} RUB")
        else:
            self.phase(f"REQUEST FAILED: model POST {'may have been sent' if sent else 'not sent' if record.get('request_sent') is False else 'status unknown'}; "
                       f"cost {'unknown' if cost is None else str(cost) + ' RUB'}; "
                       f"saved successes retained; {self.summary()}")

    def cache_progress(self, cursor, total):
        if cursor == 1 or cursor % 25 == 0 or cursor == total:
            self._line(f"LOCAL RECEIPT CHECK {cursor}/{total}; no new model request")

    def __call__(self, event, **values):
        if event == "pacing_wait":
            self.phase(f"PACING: waiting {values['wait_seconds']:.1f}s before connection setup")
        elif event in ("preflight_connect", "preflight_head"):
            action = "connecting TCP/TLS" if event == "preflight_connect" else "checking public HEAD response"
            self.phase(f"CONNECTION {values['attempt']}/{values['max_attempts']}: {action}; "
                       f"timeout {values['timeout_seconds']}s; no model POST sent")
        elif event == "preflight_ready":
            self.phase(f"CONNECTION READY: attempt {values['attempt']}/{values['max_attempts']} in {duration(values['elapsed_seconds'])}")
        elif event == "preflight_failed":
            self.phase(f"CONNECTION FAILED {values['attempt']}/{values['max_attempts']}: {safe(values['error_type'])}; "
                       f"stage={safe(values['stage'])}; elapsed {duration(values['elapsed_seconds'])}; no model POST sent")
        elif event == "preflight_retry_wait":
            self.phase(f"SAFE CONNECTION RETRY: waiting {values['wait_seconds']}s before attempt {values['next_attempt']}/3")
        elif event == "preflight_exhausted":
            self.phase("CONNECTION ATTEMPTS EXHAUSTED: no model POST sent; no model charge for this failed attempt")
        elif event == "model_post_start":
            self.phase("STARTING MODEL POST; sent requests will not be automatically repeated")

    def watch_stream(self, directory):
        self.stream_directory = Path(directory)
        self.phase("MODEL REQUEST: waiting for HTTP response / first stream bytes")

    def stop_stream(self):
        self.stream_directory = None

    def heartbeat(self):
        with self.lock:
            if self.parent is None:
                self._line(f"ALIVE: {self.phase_name}; phase elapsed {duration(self.clock() - self.phase_started)}; saved totals are loading")
                return
            extra = ""
            directory = self.stream_directory
            if directory is not None:
                tape = directory / "response.sse"
                received = tape.stat().st_size if tape.exists() else 0
                header_file = directory / "response_headers.json"
                if header_file.exists():
                    headers = json.loads(header_file.read_text(encoding="utf-8"))
                    status = headers.get("status")
                    status = status if type(status) is int and 100 <= status <= 599 else "?"
                    extra = f"; HTTP {status}; received {received:,} bytes"
                else:
                    extra = "; waiting for HTTP headers; received 0 bytes"
                progress_file = directory / "progress.jsonl"
                if progress_file.exists():
                    with progress_file.open("rb") as stream:
                        stream.seek(max(0, progress_file.stat().st_size - 16384))
                        lines = stream.read().splitlines()
                    for line in reversed(lines):
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeError):
                            continue
                        for key, label in (("chunks", "chunks"), ("reasoning_chars", "reasoning chars"),
                                           ("content_chars", "answer chars"), ("tool_argument_chars", "tool-argument chars")):
                            value = event.get(key)
                            if type(value) is int and value >= 0:
                                extra += f"; {label} {value:,}"
                        break
            self._line(f"ALIVE: {self.phase_name}; phase elapsed {duration(self.clock() - self.phase_started)}{extra}; "
                       f"{self.summary()}; known total {self.known_before + self.new_known} RUB + {self.unknown} unresolved charges")

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ObservedClient(core.ContinuationClient):
    def __init__(self, *args, progress=None, **kwargs):
        self.progress = progress
        super().__init__(*args, **kwargs)

    def complete(self, messages, tools=None, operation=None, max_tokens=None, json_output=False, response_schema=None):
        cached = self.cursor < len(self.parent["accepted"])
        op = operation or self.operation
        if self.progress is not None:
            observe(self.progress.request_start, op, self.group_id, self.traveler, cached=cached)
        before = len(self.records)
        try:
            result = super().complete(messages, tools, operation, max_tokens, json_output, response_schema)
        except BaseException as error:
            if self.progress is not None and not cached and len(self.records) > before:
                observation = dict(self.records[-1])
                if getattr(error, "request_sent", None) is not None:
                    observation["request_sent"] = error.request_sent
                if observation.get("response_accepted") is not True:
                    observation["request_status"] = "failed"
                observe(self.progress.request_result, observation, cached=False)
            raise
        if self.progress is not None:
            if len(self.records) > before:
                choice = (result.get("choices") or [{}])[0]
                reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                observe(self.progress.request_result, self.records[-1], cached=cached, finish_reason=reason)
            if cached:
                observe(self.progress.cache_progress, self.cursor, len(self.parent["accepted"]))
        return result


class ObservedTransport:
    def __init__(self, recorded, progress):
        self.recorded, self.progress = recorded, progress

    def __call__(self, method, path, payload):
        directory = self.recorded.destination / "new_requests" / f"attempt{self.recorded.attempts + 1:05d}"
        observe(self.progress.watch_stream, directory)
        try:
            return self.recorded(method, path, payload)
        finally:
            observe(self.progress.stop_stream)


def observe_session(session, progress, client):
    native_arrival = session.run_arrival
    def run_arrival(packet):
        started = observed_clock(progress)
        replay = client.cursor < len(client.parent["accepted"])
        observe(progress.arrival_start, session.scope, packet["traveler_index"], packet["name"], replay=replay)
        result = native_arrival(packet)
        observe(progress.arrival_done, session.scope, packet["traveler_index"], packet["name"],
                max(0, observed_clock(progress) - started), replayed=replay)
        return result
    session.run_arrival = run_arrival
    return session
