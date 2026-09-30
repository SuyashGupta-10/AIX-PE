"""SQLite persistence: callers, cases, transcripts, escalations, follow-ups."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta

from . import config

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    phone TEXT PRIMARY KEY, name TEXT, child_age REAL, gender TEXT, village TEXT, language TEXT,
    created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT UNIQUE, phone TEXT, kind TEXT DEFAULT 'intake',
    parent_case_id INTEGER, status TEXT DEFAULT 'open', triage_level TEXT, state_json TEXT, summary TEXT,
    created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, role TEXT, text TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, level TEXT, phc_name TEXT, phc_phone TEXT,
    summary TEXT, status TEXT DEFAULT 'pending', created_at TEXT
);
CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, phone TEXT, due_at TEXT,
    status TEXT DEFAULT 'scheduled', created_at TEXT
);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
    return _conn


def _exec(sql: str, args=()) -> sqlite3.Cursor:
    with _lock:
        cur = conn().execute(sql, args)
        conn().commit()
        return cur


def _rows(sql: str, args=()) -> list[dict]:
    with _lock:
        return [dict(r) for r in conn().execute(sql, args).fetchall()]


# --- users -----------------------------------------------------------------
def get_user(phone: str) -> dict | None:
    r = _rows("SELECT * FROM users WHERE phone=?", (phone,))
    return r[0] if r else None


def upsert_user(phone: str, **fields) -> None:
    fields = {k: v for k, v in fields.items() if v not in (None, "")}
    if get_user(phone):
        if fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            _exec(f"UPDATE users SET {sets}, updated_at=? WHERE phone=?", (*fields.values(), now(), phone))
    else:
        cols = ["phone", *fields, "created_at", "updated_at"]
        _exec(f"INSERT INTO users ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
              (phone, *fields.values(), now(), now()))


# --- cases -----------------------------------------------------------------
def create_case(session_id: str, kind: str = "intake", phone: str | None = None, parent_case_id: int | None = None) -> int:
    cur = _exec("INSERT INTO cases (session_id, phone, kind, parent_case_id, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (session_id, phone, kind, parent_case_id, now(), now()))
    return cur.lastrowid


def save_case(case_id: int, state: dict, **fields) -> None:
    fields["state_json"] = json.dumps(state, ensure_ascii=False)
    sets = ", ".join(f"{k}=?" for k in fields)
    _exec(f"UPDATE cases SET {sets}, updated_at=? WHERE id=?", (*fields.values(), now(), case_id))


def get_case(case_id: int) -> dict | None:
    r = _rows("SELECT * FROM cases WHERE id=?", (case_id,))
    return r[0] if r else None


def latest_case_for(phone: str) -> dict | None:
    r = _rows("SELECT * FROM cases WHERE phone=? AND kind='intake' AND triage_level IS NOT NULL "
              "ORDER BY id DESC LIMIT 1", (phone,))
    return r[0] if r else None


def cases_for(phone: str) -> list[dict]:
    return _rows("SELECT id, kind, status, triage_level, summary, created_at FROM cases WHERE phone=? ORDER BY id DESC",
                 (phone,))


def add_message(case_id: int, role: str, text: str) -> None:
    _exec("INSERT INTO messages (case_id, role, text, created_at) VALUES (?,?,?,?)", (case_id, role, text, now()))


def transcript(case_id: int) -> list[dict]:
    return _rows("SELECT role, text, created_at FROM messages WHERE case_id=? ORDER BY id", (case_id,))


# --- routing ---------------------------------------------------------------
def add_escalation(case_id: int, level: str, phc: dict, summary: str) -> int:
    existing = _rows("SELECT id FROM escalations WHERE case_id=? AND status='pending'", (case_id,))
    if existing:
        _exec("UPDATE escalations SET level=?, phc_name=?, phc_phone=?, summary=? WHERE id=?",
              (level, phc["name"], phc["phone"], summary, existing[0]["id"]))
        return existing[0]["id"]
    return _exec("INSERT INTO escalations (case_id, level, phc_name, phc_phone, summary, created_at) VALUES (?,?,?,?,?,?)",
                 (case_id, level, phc["name"], phc["phone"], summary, now())).lastrowid


def schedule_followup(case_id: int, phone: str | None, hours: int) -> str:
    due = (datetime.now() + timedelta(hours=hours)).isoformat(timespec="minutes")
    _exec("UPDATE followups SET status='superseded' WHERE case_id=? AND status='scheduled'", (case_id,))
    _exec("INSERT INTO followups (case_id, phone, due_at, created_at) VALUES (?,?,?,?)", (case_id, phone, due, now()))
    return due


def complete_followups(case_id: int) -> None:
    _exec("UPDATE followups SET status='done' WHERE case_id=? AND status='scheduled'", (case_id,))


def dashboard() -> dict:
    return {
        "escalations": _rows(
            "SELECT e.*, c.phone, c.session_id FROM escalations e JOIN cases c ON c.id=e.case_id "
            "ORDER BY CASE e.status WHEN 'pending' THEN 0 ELSE 1 END, e.id DESC LIMIT 50"),
        "followups": _rows(
            "SELECT f.*, c.summary, c.triage_level FROM followups f JOIN cases c ON c.id=f.case_id "
            "WHERE f.status='scheduled' ORDER BY f.due_at LIMIT 50"),
        "recent_cases": _rows(
            "SELECT id, session_id, phone, kind, status, triage_level, summary, updated_at FROM cases "
            "WHERE triage_level IS NOT NULL ORDER BY id DESC LIMIT 50"),
    }


def ack_escalation(esc_id: int) -> None:
    _exec("UPDATE escalations SET status='acknowledged' WHERE id=?", (esc_id,))


# --- PHC directory (sample data) --------------------------------------------
def find_phc(village: str | None) -> dict:
    try:
        entries = json.loads(config.PHC_DIRECTORY.read_text(encoding="utf-8"))
    except FileNotFoundError:
        entries = []
    v = (village or "").lower()
    for e in entries:
        if any(k in v for k in e["match"]):
            return e
    return {"name": "Nazdeeki PHC / CHC (district helpline)", "phone": config.EMERGENCY_NUMBER, "match": []}
