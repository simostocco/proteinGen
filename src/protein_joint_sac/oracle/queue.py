"""Single-worker durable queue, immutable identities and distinct prediction/geometry cache."""

from dataclasses import asdict
import sqlite3, json
from ..contracts import d_path, OracleStatus, OracleResult, GeneratedProtein


class EvaluationQueue:
    def __init__(self, path, max_requests=16, max_calls=8):
        self.db = sqlite3.connect(d_path(path), isolation_level=None, timeout=5)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS requests(episode TEXT PRIMARY KEY,sequence_sha TEXT,geometry_sha TEXT,protocol_sha TEXT,seed INTEGER,status TEXT,result TEXT,attempts INTEGER DEFAULT 0,evidence TEXT); CREATE TABLE IF NOT EXISTS predictions(cache_key TEXT PRIMARY KEY,result TEXT); CREATE TABLE IF NOT EXISTS budget(id INTEGER PRIMARY KEY CHECK(id=1),used INTEGER NOT NULL); INSERT OR IGNORE INTO budget VALUES(1,0);"
        )
        if type(max_requests) is not int or type(max_calls) is not int or max_requests < 1 or max_calls < 0:
            self.db.close()
            raise ValueError("valid explicit queue/call budget required")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS frozen_budget(id INTEGER PRIMARY KEY CHECK(id=1), max_requests INTEGER NOT NULL, max_calls INTEGER NOT NULL)"
        )
        self.db.execute("INSERT OR IGNORE INTO frozen_budget VALUES(1,?,?)", (max_requests, max_calls))
        frozen = self.db.execute("SELECT max_requests,max_calls FROM frozen_budget WHERE id=1").fetchone()
        if frozen != (max_requests, max_calls):
            self.db.close()
            raise ValueError("frozen oracle budget changed on resume; create separately approved run")
        self.max_requests = max_requests
        self.max_calls = max_calls

    def key(self, seq, protocol, seed):
        return seq + ":" + protocol + ":" + str(seed)

    def enqueue(self, episode, protein, protocol, seed):
        if type(protein) != GeneratedProtein:
            raise ValueError("complete sequence required")
        if seed not in protocol.seeds:
            raise ValueError("oracle seed")
        identity = (protein.sequence_sha256, protein.geometry_sha256, protocol.protocol_sha256, seed)
        evidence = "MOCK_UNIT_TEST" if protocol.scope == "SYNTHETIC_ONLY" else "EXTERNAL_PREDICTION"
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute(
                "SELECT sequence_sha,geometry_sha,protocol_sha,seed FROM requests WHERE episode=?", (episode,)
            ).fetchone()
            if old:
                if old != identity:
                    raise ValueError("immutable episode conflict")
                self.db.execute("COMMIT")
                return
            if (
                self.db.execute("SELECT count(*) FROM requests WHERE status IN ('PENDING','RUNNING')").fetchone()[0]
                >= self.max_requests
            ):
                raise ValueError("queue cap")
            cached = self.db.execute(
                "SELECT result FROM predictions WHERE cache_key=?", (self.key(identity[0], identity[2], seed),)
            ).fetchone()
            self.db.execute(
                "INSERT INTO requests(episode,sequence_sha,geometry_sha,protocol_sha,seed,status,result,evidence) VALUES(?,?,?,?,?,?,?,?)",
                (episode, *identity, "SUCCEEDED" if cached else "PENDING", cached[0] if cached else None, evidence),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def claim(self, episode):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            r = self.row(episode)
            if r["status"] != "PENDING":
                raise ValueError("only pending claim")
            cached = self.db.execute(
                "SELECT result FROM predictions WHERE cache_key=?",
                (self.key(r["sequence_sha"], r["protocol_sha"], r["seed"]),),
            ).fetchone()
            if cached:
                self.db.execute("UPDATE requests SET status='SUCCEEDED',result=? WHERE episode=?", (cached[0], episode))
                self.db.execute("COMMIT")
                return False
            if self.calls_used >= self.max_calls:
                raise ValueError("hard oracle call cap")
            if self.db.execute("SELECT count(*) FROM requests WHERE status='RUNNING'").fetchone()[0]:
                raise ValueError("one oracle worker")
            self.db.execute("UPDATE requests SET status='RUNNING',attempts=attempts+1 WHERE episode=?", (episode,))
            self.db.execute("UPDATE budget SET used=used+1 WHERE id=1")
            self.db.execute("COMMIT")
            return True
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def finish(self, episode, result):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            r = self.row(episode)
            if r["status"] != "RUNNING" or (r["sequence_sha"], r["protocol_sha"], r["seed"], r["evidence"]) != (
                result.sequence_sha256,
                result.protocol_sha256,
                result.seed,
                result.evidence_type,
            ):
                raise ValueError("result identity/state mismatch")
            if result.status not in [
                OracleStatus.SUCCEEDED,
                OracleStatus.FAILED,
                OracleStatus.TIMEOUT,
                OracleStatus.UNSUPPORTED,
            ]:
                raise ValueError("terminal result required")
            payload = json.dumps(asdict(result), sort_keys=True, allow_nan=False)
            if result.status == OracleStatus.SUCCEEDED:
                key = self.key(r["sequence_sha"], r["protocol_sha"], r["seed"])
                old = self.db.execute("SELECT result FROM predictions WHERE cache_key=?", (key,)).fetchone()
                if old and old[0] != payload:
                    raise ValueError("immutable prediction cache")
                self.db.execute("INSERT OR IGNORE INTO predictions VALUES(?,?)", (key, payload))
            self.db.execute("UPDATE requests SET status=?,result=? WHERE episode=?", (result.status, payload, episode))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def row(self, episode):
        c = self.db.execute(
            "SELECT episode,sequence_sha,geometry_sha,protocol_sha,seed,status,result,attempts,evidence FROM requests WHERE episode=?",
            (episode,),
        )
        r = c.fetchone()
        if r is None:
            raise KeyError(episode)
        return dict(zip([x[0] for x in c.description], r))

    def recover_running(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.db.execute(
                "SELECT episode,sequence_sha,protocol_sha,seed,evidence FROM requests WHERE status='RUNNING'"
            ).fetchall()
            for episode, seq, protocol, seed, evidence in rows:
                result = OracleResult(
                    seq,
                    protocol,
                    OracleStatus.TIMEOUT,
                    seed,
                    evidence,
                    failure_reason="interrupted worker; consumed call budget preserved",
                )
                self.db.execute(
                    "UPDATE requests SET status='TIMEOUT',result=? WHERE episode=?",
                    (json.dumps(asdict(result)), episode),
                )
            self.db.execute("COMMIT")
            return len(rows)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    @property
    def calls_used(self):
        return self.db.execute("SELECT used FROM budget WHERE id=1").fetchone()[0]

    def close(self):
        self.db.close()
