import contextlib
import datetime as dt
import json
import sqlite3
import threading
import uuid
from pathlib import Path


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def identity(prefix):
    return prefix + "_" + uuid.uuid4().hex[:16]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,name TEXT,settings TEXT,graph TEXT,version INTEGER,status TEXT,created_at TEXT);
                CREATE TABLE IF NOT EXISTS nodes(run_id TEXT,id TEXT,data TEXT,PRIMARY KEY(run_id,id));
                CREATE TABLE IF NOT EXISTS links(run_id TEXT,id TEXT,data TEXT,PRIMARY KEY(run_id,id));
                CREATE TABLE IF NOT EXISTS hyperedges(run_id TEXT,id TEXT,data TEXT,PRIMARY KEY(run_id,id));
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE,run_id TEXT,kind TEXT,data TEXT,created_at TEXT);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,run_id TEXT,key TEXT,request TEXT,status TEXT,error TEXT,result TEXT,created_at TEXT,UNIQUE(run_id,key));
                CREATE UNIQUE INDEX IF NOT EXISTS active_job ON jobs(run_id) WHERE status IN ('queued','running');
                CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY,run_id TEXT,data TEXT);
                CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,data TEXT);
                CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,run_id TEXT,job_id TEXT,data TEXT);
                CREATE TABLE IF NOT EXISTS model_cache(key TEXT PRIMARY KEY,data TEXT);
                CREATE TABLE IF NOT EXISTS benchmark_sessions(run_id TEXT PRIMARY KEY,data TEXT,control TEXT);
                CREATE TRIGGER IF NOT EXISTS immutable_events_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'Events are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_events_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'Events are immutable'); END;
            """)

    @contextlib.contextmanager
    def connection(self):
        with self.lock:
            db = sqlite3.connect(self.path, timeout=15)
            db.row_factory = sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()

    def event(self, db, run_id, kind, data, event_id=None):
        event_id = event_id or identity("event")
        db.execute("INSERT INTO events(id,run_id,kind,data,created_at) VALUES(?,?,?,?,?)", (event_id, run_id, kind, encoded(data), now()))
        return event_id

    def create(self, name, settings):
        run_id = identity("run")
        graph = {"nodes": [], "links": [], "hyperedges": [], "projection": None, "graph_version": 0}
        with self.connection() as db:
            db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?)", (run_id, name, encoded(settings), encoded(graph), 0, "ready", now()))
            self.event(db, run_id, "run.created", {"name": name, "settings": settings, "graph": graph})
        return self.run(run_id)

    def run(self, run_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("Run not found")
        result = dict(row)
        for key in ("settings", "graph"):
            result[key] = json.loads(result[key])
        return result

    def list_runs(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT id,name,version,status,created_at FROM runs ORDER BY created_at DESC")]

    def events(self, run_id, after=0):
        with self.connection() as db:
            return [{**dict(row), "data": json.loads(row["data"])} for row in db.execute("SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after))]

    def reserve(self, run_id, action):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM jobs WHERE run_id=? AND key=?", (run_id, action.idempotency_key)).fetchone()
            if old:
                if old["request"] != encoded(action.model_dump()):
                    raise Conflict("Idempotency key already belongs to a different request")
                return self.job(run_id, old["id"]), False
            run = self.run(run_id)
            if run["version"] != action.expected_graph_version:
                raise Conflict("Graph changed; refresh before starting this action")
            if db.execute("SELECT 1 FROM jobs WHERE run_id=? AND status IN ('queued','running')", (run_id,)).fetchone():
                raise Conflict("A mutating job is already active for this run")
            job_id = identity("job")
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)", (job_id, run_id, action.idempotency_key, encoded(action.model_dump()), "queued", None, None, now()))
            db.execute("UPDATE runs SET status='action_running' WHERE id=?", (run_id,))
            self.event(db, run_id, "job.created", {"job_id": job_id, "action": action.action})
        return self.job(run_id, job_id), True

    def job(self, run_id, job_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=? AND run_id=?", (job_id, run_id)).fetchone()
        if row is None:
            raise KeyError("Job not found")
        return {**dict(row), "request": json.loads(row["request"]), "result": json.loads(row["result"]) if row["result"] else None}

    def transition(self, run_id, job_id, status, error=None):
        with self.connection() as db:
            job = db.execute("SELECT status FROM jobs WHERE id=? AND run_id=?", (job_id, run_id)).fetchone()
            if not job:
                raise KeyError("Job not found")
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                return False
            db.execute("UPDATE jobs SET status=?,error=? WHERE id=?", (status, error, job_id))
            if status in {"failed", "cancelled"}:
                db.execute("UPDATE runs SET status=? WHERE id=?", ("action_failed" if status == "failed" else "ready", run_id))
            self.event(db, run_id, "job." + status, {"job_id": job_id, "error": error})
        return True

    def commit(self, run_id, job_id, expected, graph, result, event_id, *, final=True):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job["status"] != "running":
                return False
            if self.run(run_id)["version"] != expected:
                raise Conflict("Graph changed during operation")
            graph["graph_version"] = expected + 1
            db.execute("UPDATE runs SET graph=?,version=?,status=? WHERE id=?", (encoded(graph), expected + 1, "ready" if final else "action_running", run_id))
            for table, key in (("nodes", "nodes"), ("links", "links"), ("hyperedges", "hyperedges")):
                for obj in graph[key]:
                    db.execute(f"INSERT OR REPLACE INTO {table} VALUES(?,?,?)", (run_id, obj["id"], encoded(obj)))
            self.event(db, run_id, "graph.patch", {"graph_version": expected + 1, "graph": graph, "result": result, "job_id": job_id}, event_id)
            if final:
                db.execute("UPDATE jobs SET status='succeeded',error=NULL,result=? WHERE id=?", (encoded(result), job_id))
                self.event(db, run_id, "job.succeeded", {"job_id": job_id, "result": result})
            else:
                db.execute("UPDATE jobs SET result=? WHERE id=?", (encoded(result), job_id))
        return True

    def receipt(self, run_id, job_id, data):
        with self.connection() as db:
            db.execute("INSERT INTO calls VALUES(?,?,?,?)", (identity("call"), run_id, job_id, encoded(data)))

    def receipts(self, run_id):
        with self.connection() as db:
            return [{**dict(row), "data": json.loads(row["data"])} for row in db.execute("SELECT * FROM calls WHERE run_id=?", (run_id,))]

    def jobs(self, run_id):
        self.run(run_id)
        with self.connection() as db:
            return [self.job(run_id, row["id"]) for row in db.execute("SELECT id FROM jobs WHERE run_id=? ORDER BY created_at", (run_id,))]

    def recover(self):
        with self.connection() as db:
            unfinished = [dict(row) for row in db.execute("SELECT id,run_id FROM jobs WHERE status IN ('queued','running')")]
        for row in unfinished:
            self.transition(row["run_id"], row["id"], "failed", "Server restarted before completion; model execution may have occurred. Graph was not committed. Retry explicitly.")
