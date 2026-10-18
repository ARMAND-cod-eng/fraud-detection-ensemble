"""SQLite persistence for predictions, decisions, analyst case actions and policy history.

SQLite (WAL mode) is sufficient for a single-instance demo and keeps local setup to zero
services. The interface is narrow so it can be swapped for Postgres in a multi-instance
deployment (see docs/ENGINEERING_TRADEOFFS.md).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    transaction_id    TEXT PRIMARY KEY,
    payload_hash      TEXT NOT NULL,
    received_at       REAL NOT NULL,
    event_time        REAL,
    amount            REAL NOT NULL,
    fraud_probability REAL NOT NULL,
    raw_score         REAL NOT NULL,
    risk_level        TEXT NOT NULL,
    action            TEXT NOT NULL,
    reasons           TEXT NOT NULL,
    expected_loss     REAL NOT NULL,
    over_capacity     INTEGER NOT NULL DEFAULT 0,
    model_version     TEXT NOT NULL,
    policy_version    TEXT NOT NULL,
    latency_ms        REAL NOT NULL,
    source            TEXT NOT NULL,
    synthetic         INTEGER NOT NULL DEFAULT 0,
    scenario          TEXT,
    ground_truth      INTEGER,
    features          TEXT NOT NULL,
    explanation       TEXT,
    signals           TEXT,
    quality           TEXT NOT NULL,
    review_status     TEXT NOT NULL,
    top_reasons       TEXT
);
CREATE INDEX IF NOT EXISTS ix_pred_received ON predictions(received_at);
CREATE INDEX IF NOT EXISTS ix_pred_action ON predictions(action, received_at);
CREATE INDEX IF NOT EXISTS ix_pred_review ON predictions(review_status, received_at);

CREATE TABLE IF NOT EXISTS case_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL REFERENCES predictions(transaction_id),
    at             REAL NOT NULL,
    actor          TEXT NOT NULL,
    event          TEXT NOT NULL,
    decision       TEXT,
    notes          TEXT
);
CREATE INDEX IF NOT EXISTS ix_case_txn ON case_events(transaction_id, at);

-- Linked activity: transactions sharing an entity (card, card+e-mail, device model...). Keys are
-- SHA-256 prefixes of the entity values: an index key, not anonymisation (low-entropy values can be
-- brute-forced), so this table is as sensitive as the features it is derived from.
CREATE TABLE IF NOT EXISTS entity_links (
    entity_type    TEXT NOT NULL,
    entity_hash    TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    event_time     REAL,
    PRIMARY KEY (entity_type, entity_hash, transaction_id)
);
CREATE INDEX IF NOT EXISTS ix_links_txn ON entity_links(transaction_id);
CREATE INDEX IF NOT EXISTS ix_links_entity_time ON entity_links(entity_type, entity_hash, event_time);

CREATE TABLE IF NOT EXISTS policy_history (
    version TEXT NOT NULL,
    at      REAL NOT NULL,
    actor   TEXT NOT NULL,
    config  TEXT NOT NULL
);
"""

JSON_COLUMNS = ("reasons", "features", "explanation", "signals", "quality", "top_reasons")
# Columns added after the first release; ``_migrate`` adds them to older local databases.
ADDED_COLUMNS = {"top_reasons": "TEXT"}
REVIEW_DECISIONS = {"confirmed_fraud", "legitimate", "escalated"}


class PredictionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        have = {r[1] for r in self._conn.execute("PRAGMA table_info(predictions)").fetchall()}
        for col, sql_type in ADDED_COLUMNS.items():
            if col not in have:
                self._conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {sql_type}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def ping(self) -> bool:
        try:
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    # ------------------------------------------------------------------ writes
    def get_hash(self, transaction_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_hash FROM predictions WHERE transaction_id = ?", (transaction_id,)
            ).fetchone()
        return None if row is None else str(row[0])

    def insert_predictions(
        self,
        rows: Iterable[dict[str, Any]],
        links: Iterable[tuple[str, str, str, float]] = (),
    ) -> None:
        """Insert predictions and their entity links in one transaction."""
        rows = list(rows)
        if not rows:
            return
        links = list(links)
        cols = list(rows[0].keys())
        sql = f"INSERT INTO predictions ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        values = [
            tuple(json.dumps(r[c]) if c in JSON_COLUMNS and r[c] is not None else r[c] for c in cols)
            for r in rows
        ]
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(sql, values)
                if links:
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO entity_links (entity_type, entity_hash, transaction_id, event_time) "
                        "VALUES (?, ?, ?, ?)",
                        links,
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def set_explanation(
        self, transaction_id: str, explanation: dict[str, Any], signals: dict[str, Any]
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE predictions SET explanation = ?, signals = ? WHERE transaction_id = ?",
                (json.dumps(explanation), json.dumps(signals), transaction_id),
            )

    def add_case_event(
        self, transaction_id: str, actor: str, event: str, decision: str | None, notes: str | None
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                cur = self._conn.execute(
                    "INSERT INTO case_events (transaction_id, at, actor, event, decision, notes) VALUES (?, ?, ?, ?, ?, ?)",
                    (transaction_id, now, actor, event, decision, notes),
                )
                if decision in REVIEW_DECISIONS:
                    self._conn.execute(
                        "UPDATE predictions SET review_status = ? WHERE transaction_id = ?",
                        (decision, transaction_id),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return {
            "id": cur.lastrowid,
            "transaction_id": transaction_id,
            "at": now,
            "actor": actor,
            "event": event,
            "decision": decision,
            "notes": notes,
        }

    def add_policy_version(self, version: str, actor: str, config: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO policy_history (version, at, actor, config) VALUES (?, ?, ?, ?)",
                (version, time.time(), actor, json.dumps(config)),
            )

    def prune(self, keep: int) -> int:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            if n <= keep:
                return 0
            cutoff = self._conn.execute(
                "SELECT received_at FROM predictions ORDER BY received_at DESC LIMIT 1 OFFSET ?", (keep,)
            ).fetchone()[0]
            cur = self._conn.execute(
                "DELETE FROM predictions WHERE received_at <= ? AND review_status IN ('none', 'open')",
                (cutoff,),
            )
            self._conn.execute(
                "DELETE FROM entity_links WHERE transaction_id NOT IN (SELECT transaction_id FROM predictions)"
            )
            return cur.rowcount

    def reset(self) -> None:
        with self._lock:
            self._conn.executescript(
                "DELETE FROM case_events; DELETE FROM entity_links; DELETE FROM predictions;"
            )

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for c in JSON_COLUMNS:
            if c in d and d[c] is not None:
                d[c] = json.loads(d[c])
        for c in ("synthetic", "over_capacity"):
            if c in d:
                d[c] = bool(d[c])
        return d

    def get(self, transaction_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM predictions WHERE transaction_id = ?", (transaction_id,)
            ).fetchone()
        return None if row is None else self._decode(row)

    def case_events(self, transaction_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM case_events WHERE transaction_id = ? ORDER BY at", (transaction_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def list_predictions(
        self,
        limit: int = 100,
        offset: int = 0,
        action: str | None = None,
        risk_level: str | None = None,
        review_status: str | None = None,
        alerts_only: bool = False,
        search: str | None = None,
        min_probability: float | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        where: list[str] = []
        args: list[Any] = []
        if action:
            where.append("action = ?")
            args.append(action)
        if risk_level:
            where.append("risk_level = ?")
            args.append(risk_level)
        if review_status:
            where.append("review_status = ?")
            args.append(review_status)
        if alerts_only:
            where.append("action != 'approve'")
        if search:
            where.append("transaction_id LIKE ?")
            args.append(f"%{search}%")
        if min_probability is not None:
            where.append("fraud_probability >= ?")
            args.append(min_probability)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        cols = (
            "transaction_id, received_at, event_time, amount, fraud_probability, risk_level, action, reasons, "
            "model_version, policy_version, latency_ms, source, synthetic, scenario, review_status, over_capacity, "
            "quality, top_reasons, explanation IS NOT NULL AS has_explanation"
        )
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM predictions {clause}", args).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT {cols} FROM predictions {clause} ORDER BY received_at DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        out = []
        for r in rows:
            d = self._decode(r)
            d["has_explanation"] = bool(d["has_explanation"])
            out.append(d)
        return out, int(total)

    def overview(self, since: float | None = None) -> dict[str, Any]:
        clause, args = ("WHERE received_at >= ?", [since]) if since else ("", [])
        with self._lock:
            row = self._conn.execute(
                f"""SELECT COUNT(*) n,
                    SUM(action != 'approve') alerts,
                    SUM(action = 'review') reviews,
                    SUM(action = 'decline') declines,
                    COALESCE(SUM(amount), 0) volume,
                    COALESCE(SUM(fraud_probability * amount), 0) expected_fraud_amount,
                    COALESCE(SUM(CASE WHEN action != 'approve' THEN amount END), 0) flagged_amount,
                    SUM(review_status = 'open') open_cases,
                    SUM(review_status = 'confirmed_fraud') confirmed_fraud,
                    SUM(review_status = 'legitimate') confirmed_legit,
                    SUM(synthetic) synthetic,
                    MIN(received_at) first_at, MAX(received_at) last_at
                FROM predictions {clause}""",
                args,
            ).fetchone()
            risk = self._conn.execute(
                f"SELECT risk_level, COUNT(*) FROM predictions {clause} GROUP BY risk_level", args
            ).fetchall()
            hist = self._conn.execute(
                f"""SELECT CASE
                      WHEN fraud_probability < 0.001 THEN 0 WHEN fraud_probability < 0.01 THEN 1
                      WHEN fraud_probability < 0.05 THEN 2 WHEN fraud_probability < 0.1 THEN 3
                      WHEN fraud_probability < 0.25 THEN 4 WHEN fraud_probability < 0.5 THEN 5
                      WHEN fraud_probability < 0.75 THEN 6 WHEN fraud_probability < 0.9 THEN 7 ELSE 8 END b,
                    COUNT(*) FROM predictions {clause} GROUP BY b""",
                args,
            ).fetchall()
        d = dict(row)
        d = {k: (v or 0) for k, v in d.items()}
        d["alert_rate"] = d["alerts"] / d["n"] if d["n"] else 0.0
        d["risk_distribution"] = {k: int(v) for k, v in risk}
        edges = ["0", "0.1%", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "100%"]
        counts = {int(b): int(c) for b, c in hist}
        d["probability_histogram"] = [
            {"bin": f"{edges[i]}–{edges[i + 1]}", "count": counts.get(i, 0)} for i in range(9)
        ]
        return d

    def timeseries(self, bucket_s: int = 10, points: int = 60) -> list[dict[str, Any]]:
        now = time.time()
        start = (int(now) // bucket_s - points + 1) * bucket_s
        with self._lock:
            rows = self._conn.execute(
                """SELECT CAST(received_at / ? AS INTEGER) * ? AS t, COUNT(*) n, SUM(action != 'approve') alerts,
                          AVG(latency_ms) latency
                   FROM predictions WHERE received_at >= ? GROUP BY t ORDER BY t""",
                (bucket_s, bucket_s, start),
            ).fetchall()
        by_t = {int(r["t"]): r for r in rows}
        out = []
        for i in range(points):
            t = start + i * bucket_s
            r = by_t.get(t)
            out.append(
                {
                    "t": t,
                    "transactions": int(r["n"]) if r else 0,
                    "alerts": int(r["alerts"] or 0) if r else 0,
                    "avg_latency_ms": float(r["latency"]) if r and r["latency"] is not None else None,
                }
            )
        return out

    def linked(self, transaction_id: str, event_time: float | None, limit: int = 10) -> dict[str, Any]:
        """Per entity type: transactions sharing the key, with counts before/after this event."""
        out: dict[str, Any] = {}
        with self._lock:
            keys = self._conn.execute(
                "SELECT entity_type, entity_hash FROM entity_links WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchall()
            for kind, h in keys:
                counts = self._conn.execute(
                    """SELECT COUNT(*) total,
                              SUM(event_time < ?) earlier,
                              SUM(event_time >= ? - 86400 AND event_time < ?) earlier_24h
                       FROM entity_links WHERE entity_type = ? AND entity_hash = ? AND transaction_id != ?""",
                    (event_time, event_time, event_time, kind, h, transaction_id),
                ).fetchone()
                items = self._conn.execute(
                    """SELECT p.transaction_id, p.event_time, p.amount, p.fraud_probability, p.action,
                              p.risk_level, p.review_status, p.received_at
                       FROM entity_links l JOIN predictions p ON p.transaction_id = l.transaction_id
                       WHERE l.entity_type = ? AND l.entity_hash = ? AND l.transaction_id != ?
                       ORDER BY ABS(l.event_time - ?) LIMIT ?""",
                    (kind, h, transaction_id, event_time, limit),
                ).fetchall()
                out[kind] = {
                    "total": int(counts["total"] or 0),
                    "earlier": int(counts["earlier"] or 0),
                    "earlier_24h": int(counts["earlier_24h"] or 0),
                    "transactions": sorted((dict(r) for r in items), key=lambda d: d["event_time"] or 0),
                }
        return out

    def labelled_outcomes(self, limit: int = 20_000) -> list[tuple[float, int, str]]:
        """(probability, ground_truth, action) for replayed historical rows with known labels."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT fraud_probability, ground_truth, action FROM predictions WHERE ground_truth IS NOT NULL "
                "ORDER BY received_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(float(r[0]), int(r[1]), str(r[2])) for r in rows]

    def analyst_feedback_outcomes(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT review_status, COUNT(*) FROM predictions WHERE review_status IN ('confirmed_fraud', 'legitimate') "
                "GROUP BY review_status"
            ).fetchall()
        return {str(k): int(v) for k, v in rows}
