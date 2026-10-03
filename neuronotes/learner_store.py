"""learner_store.py — Persistent state and warm-start for multi-user adaptive testing."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set, List, Dict, Any
import numpy as np


@dataclass
class LearnerRecord:
    learner_id: str
    theta: np.ndarray
    concept_theta: Dict[str, float]
    misconception_probs: np.ndarray
    session_count: int
    last_session_time: datetime
    total_questions_answered: int
    dim: int


class LearnerStore:
    """SQLite-backed multi-user learner session store with time-decayed warm starts."""

    def __init__(self, db_path: str | Path = "Neuronotes/results/learner_store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS learners (
                    learner_id TEXT PRIMARY KEY,
                    theta BLOB NOT NULL,
                    concept_theta TEXT,
                    misconception_probs BLOB,
                    session_count INTEGER DEFAULT 1,
                    last_session_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_questions_answered INTEGER DEFAULT 0,
                    dim INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    session_start TIMESTAMP,
                    session_end TIMESTAMP,
                    num_questions INTEGER,
                    accuracy REAL,
                    FOREIGN KEY(learner_id) REFERENCES learners(learner_id)
                );

                CREATE TABLE IF NOT EXISTS question_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    concept TEXT,
                    correct INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_qh_learner ON question_history(learner_id);
                CREATE INDEX IF NOT EXISTS idx_qh_item ON question_history(learner_id, item_id);
                """
            )

    def get_learner(self, learner_id: str) -> Optional[LearnerRecord]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT learner_id, theta, concept_theta, misconception_probs,
                   session_count, last_session_time, total_questions_answered, dim
            FROM learners
            WHERE learner_id = ?
            """,
            (learner_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        (lid, theta_blob, concept_theta_json, misc_blob,
         sess_cnt, last_time, total_q, dim) = row

        theta = np.frombuffer(theta_blob, dtype=np.float64)
        concept_theta = json.loads(concept_theta_json) if concept_theta_json else {}
        misc_probs = (
            np.frombuffer(misc_blob, dtype=np.float64)
            if misc_blob
            else np.array([], dtype=np.float64)
        )

        if isinstance(last_time, str):
            try:
                last_time = datetime.fromisoformat(last_time)
            except ValueError:
                last_time = datetime.now(timezone.utc)
        elif not isinstance(last_time, datetime):
            last_time = datetime.now(timezone.utc)

        return LearnerRecord(
            learner_id=lid,
            theta=theta,
            concept_theta=concept_theta,
            misconception_probs=misc_probs,
            session_count=sess_cnt,
            last_session_time=last_time,
            total_questions_answered=total_q,
            dim=dim,
        )

    def get_warm_start_theta(
        self,
        learner_id: str,
        theta_prior: np.ndarray,
        lambda_decay: float = 0.05,
        current_time: Optional[datetime] = None,
    ) -> Optional[np.ndarray]:
        """Compute time-decayed ability vector for next session.

        Formula:
            theta_warm = theta_prior + (theta_last - theta_prior) * exp(-lambda * delta_days)
        """
        record = self.get_learner(learner_id)
        if record is None:
            return None

        if len(record.theta) != len(theta_prior):
            # Dimension mismatch: model dimension changed, fallback gracefully
            return None

        if current_time is None:
            current_time = datetime.now(timezone.utc)

        if record.last_session_time.tzinfo is None and current_time.tzinfo is not None:
            last_time = record.last_session_time.replace(tzinfo=timezone.utc)
        elif record.last_session_time.tzinfo is not None and current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
            last_time = record.last_session_time
        else:
            last_time = record.last_session_time

        delta_seconds = max(0.0, (current_time - last_time).total_seconds())
        delta_days = delta_seconds / 86400.0

        decay = float(np.exp(-lambda_decay * delta_days))
        theta_warm = theta_prior + (record.theta - theta_prior) * decay
        return np.asarray(theta_warm, dtype=np.float64)

    def get_seen_items(
        self, learner_id: str, max_recent_sessions: Optional[int] = None
    ) -> Set[str]:
        cur = self.conn.cursor()
        if max_recent_sessions is None:
            cur.execute(
                "SELECT DISTINCT item_id FROM question_history WHERE learner_id = ?",
                (learner_id,),
            )
        else:
            cur.execute(
                """
                SELECT DISTINCT item_id FROM question_history
                WHERE learner_id = ? AND session_id IN (
                    SELECT session_id FROM session_history
                    WHERE learner_id = ?
                    ORDER BY session_end DESC
                    LIMIT ?
                )
                """,
                (learner_id, learner_id, max_recent_sessions),
            )
        return {str(row[0]) for row in cur.fetchall()}

    def get_spaced_review_eligibility(
        self, learner_id: str, max_recent_sessions: Optional[int] = 2
    ) -> Dict[str, Set[str]]:
        """Classify past items for SpacedCAT policy:
        - wrong_items: items answered incorrectly in prior sessions (remediation eligible).
        - correct_items: items answered correctly in recent sessions (exhaustion eligible).
        - all_prior_items: union of recent seen items from past sessions.

        When max_recent_sessions is provided (default=2), only questions from the
        last N sessions are tracked. Older items (> N sessions ago) naturally
        re-enter the item bank, preventing item bank starvation across many sessions.
        """
        cur = self.conn.cursor()
        if max_recent_sessions is None:
            cur.execute(
                """
                SELECT item_id, correct
                FROM question_history
                WHERE learner_id = ?
                ORDER BY timestamp ASC
                """,
                (learner_id,),
            )
        else:
            cur.execute(
                """
                SELECT item_id, correct
                FROM question_history
                WHERE learner_id = ? AND session_id IN (
                    SELECT session_id FROM session_history
                    WHERE learner_id = ?
                    ORDER BY session_end DESC
                    LIMIT ?
                )
                ORDER BY timestamp ASC
                """,
                (learner_id, learner_id, max_recent_sessions),
            )
        history = cur.fetchall()
        wrong_items = set()
        correct_items = set()
        for item_id, correct in history:
            iid = str(item_id)
            if not correct:
                wrong_items.add(iid)
                correct_items.discard(iid)
            else:
                correct_items.add(iid)
                wrong_items.discard(iid)

        return {
            "wrong_items": wrong_items,
            "correct_items": correct_items,
            "all_prior_items": wrong_items | correct_items,
        }

    def save_learner_session(
        self,
        learner_id: str,
        session_id: str,
        theta: np.ndarray,
        concept_theta: Dict[str, float],
        misconception_probs: np.ndarray,
        questions_record: List[Dict[str, Any]],
        session_start: datetime,
        session_end: datetime,
    ) -> None:
        """Persist final learner state and question interaction history."""
        theta_arr = np.asarray(theta, dtype=np.float64)
        theta_blob = theta_arr.tobytes()
        dim = len(theta_arr)

        misc_arr = np.asarray(misconception_probs, dtype=np.float64)
        misc_blob = misc_arr.tobytes()
        concept_json = json.dumps(concept_theta)

        num_questions = len(questions_record)
        accuracy = (
            float(sum(1 for q in questions_record if q.get("correct", False)) / num_questions)
            if num_questions > 0
            else 0.0
        )

        t_start_str = session_start.isoformat() if isinstance(session_start, datetime) else str(session_start)
        t_end_str = session_end.isoformat() if isinstance(session_end, datetime) else str(session_end)

        with self.conn:
            # 1. Update or insert learner profile
            self.conn.execute(
                """
                INSERT INTO learners (
                    learner_id, theta, concept_theta, misconception_probs,
                    session_count, last_session_time, total_questions_answered, dim
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(learner_id) DO UPDATE SET
                    theta = excluded.theta,
                    concept_theta = excluded.concept_theta,
                    misconception_probs = excluded.misconception_probs,
                    session_count = learners.session_count + 1,
                    last_session_time = excluded.last_session_time,
                    total_questions_answered = learners.total_questions_answered + excluded.total_questions_answered,
                    dim = excluded.dim
                """,
                (
                    learner_id,
                    theta_blob,
                    concept_json,
                    misc_blob,
                    t_end_str,
                    num_questions,
                    dim,
                ),
            )

            # 2. Insert session entry
            self.conn.execute(
                """
                INSERT INTO session_history (
                    learner_id, session_id, session_start, session_end,
                    num_questions, accuracy
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    learner_id,
                    session_id,
                    t_start_str,
                    t_end_str,
                    num_questions,
                    accuracy,
                ),
            )

            # 3. Insert question interactions
            def _format_q_time(q_item):
                ts = q_item.get("timestamp", session_end)
                return ts.isoformat() if isinstance(ts, datetime) else str(ts)

            rows = [
                (
                    learner_id,
                    session_id,
                    str(q["item_id"]),
                    str(q.get("concept", "")),
                    1 if q.get("correct", False) else 0,
                    _format_q_time(q),
                )
                for q in questions_record
            ]
            if rows:
                self.conn.executemany(
                    """
                    INSERT INTO question_history (
                        learner_id, session_id, item_id, concept, correct, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def delete_learner(self, learner_id: str) -> bool:
        """Cascade delete all traces of a learner from database."""
        with self.conn:
            self.conn.execute("DELETE FROM question_history WHERE learner_id = ?", (learner_id,))
            self.conn.execute("DELETE FROM session_history WHERE learner_id = ?", (learner_id,))
            cur = self.conn.execute("DELETE FROM learners WHERE learner_id = ?", (learner_id,))
            return cur.rowcount > 0

    def clear_all(self) -> int:
        """Wipe all learner sessions and history from store."""
        with self.conn:
            self.conn.execute("DELETE FROM question_history")
            self.conn.execute("DELETE FROM session_history")
            cur = self.conn.execute("DELETE FROM learners")
            return cur.rowcount

    def list_learners(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List summary of saved learners."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT learner_id, session_count, total_questions_answered, last_session_time, dim
            FROM learners
            ORDER BY last_session_time DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "learner_id": row[0],
                "session_count": row[1],
                "total_questions_answered": row[2],
                "last_session_time": str(row[3]),
                "dim": row[4],
            }
            for row in cur.fetchall()
        ]

    def close(self) -> None:
        if self.conn:
            self.conn.close()

    def __enter__(self) -> LearnerStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
