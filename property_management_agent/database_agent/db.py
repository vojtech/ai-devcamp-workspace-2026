"""
SQLite schema and CRUD helpers used directly by the database_agent tools.
DB_PATH defaults to property_data.db next to this file, or from DB_PATH env var.
"""
import json
import os
import sqlite3
from datetime import datetime
from typing import Any

DB_PATH = os.getenv(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "property_data.db"),
)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS employees (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                role        TEXT,
                department  TEXT,
                phone       TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS property_managers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'current',
                properties  TEXT,
                notes       TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS key_contacts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT,
                phone       TEXT,
                role        TEXT,
                company     TEXT,
                notes       TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT NOT NULL,
                description  TEXT,
                assigned_to  TEXT,
                due_date     TEXT,
                status       TEXT DEFAULT 'open',
                priority     TEXT DEFAULT 'normal',
                source_email TEXT,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS meetings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT NOT NULL,
                date         TEXT,
                time         TEXT,
                attendees    TEXT,
                location     TEXT,
                agenda       TEXT,
                outcome      TEXT,
                source_email TEXT,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)


init_db()


# ── Employees ──────────────────────────────────────────────────────────────────

def upsert_employee(name: str, email: str, role: str = "", department: str = "", phone: str = "") -> str:
    """Insert or update an employee record. Returns a confirmation string."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO employees (name, email, role, department, phone)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name=excluded.name,
                role=COALESCE(NULLIF(excluded.role,''), role),
                department=COALESCE(NULLIF(excluded.department,''), department),
                phone=COALESCE(NULLIF(excluded.phone,''), phone)
        """, (name, email, role, department, phone))
    return f"Employee '{name}' ({email}) saved."


def list_employees() -> str:
    """Return all employees as a JSON string."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM employees ORDER BY name").fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Property Managers ──────────────────────────────────────────────────────────

def upsert_property_manager(
    name: str, email: str, status: str = "current", properties: str = "", notes: str = ""
) -> str:
    """Insert or update a property manager. status: 'current' or 'previous'."""
    if status not in ("current", "previous"):
        return f"Error: status must be 'current' or 'previous', got '{status}'"
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM property_managers WHERE email=?", (email,)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE property_managers
                SET name=?, status=?,
                    properties=COALESCE(NULLIF(?,  ''), properties),
                    notes=COALESCE(NULLIF(?,  ''), notes)
                WHERE email=?
            """, (name, status, properties, notes, email))
        else:
            conn.execute("""
                INSERT INTO property_managers (name, email, status, properties, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (name, email, status, properties, notes))
    return f"Property manager '{name}' ({status}) saved."


def list_property_managers() -> str:
    """Return all property managers as a JSON string."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM property_managers ORDER BY status DESC, name"
        ).fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Key Contacts ───────────────────────────────────────────────────────────────

def upsert_key_contact(
    name: str, role: str, email: str = "", phone: str = "", company: str = "", notes: str = ""
) -> str:
    """Insert or update a key contact (tenant, contractor, landlord, etc.)."""
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM key_contacts WHERE name=? AND role=?", (name, role)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE key_contacts
                SET email=COALESCE(NULLIF(?,  ''), email),
                    phone=COALESCE(NULLIF(?,  ''), phone),
                    company=COALESCE(NULLIF(?,  ''), company),
                    notes=COALESCE(NULLIF(?,  ''), notes)
                WHERE id=?
            """, (email, phone, company, notes, existing["id"]))
        else:
            conn.execute("""
                INSERT INTO key_contacts (name, email, phone, role, company, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, email, phone, role, company, notes))
    return f"Contact '{name}' ({role}) saved."


def list_key_contacts() -> str:
    """Return all key contacts as a JSON string."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM key_contacts ORDER BY role, name").fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Tasks ──────────────────────────────────────────────────────────────────────

def add_task(
    title: str,
    description: str = "",
    assigned_to: str = "",
    due_date: str = "",
    priority: str = "normal",
    source_email: str = "",
) -> str:
    """Add a new task. priority: 'normal' or 'high'."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO tasks (title, description, assigned_to, due_date, priority, source_email)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, description, assigned_to, due_date, priority, source_email))
    return f"Task '{title}' saved."


def update_task_status(task_id: int, status: str) -> str:
    """Update task status: 'open', 'in_progress', or 'completed'."""
    with get_conn() as conn:
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    return f"Task {task_id} updated to '{status}'."


def list_tasks(status: str = "") -> str:
    """Return tasks as JSON, optionally filtered by status."""
    with get_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY priority DESC, due_date",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY priority DESC, due_date"
            ).fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Meetings ───────────────────────────────────────────────────────────────────

def add_meeting(
    title: str,
    date: str = "",
    time: str = "",
    attendees: str = "",
    location: str = "",
    agenda: str = "",
    source_email: str = "",
) -> str:
    """Add a meeting extracted from email."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO meetings (title, date, time, attendees, location, agenda, source_email)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, date, time, attendees, location, agenda, source_email))
    return f"Meeting '{title}' saved."


def list_meetings() -> str:
    """Return all meetings as JSON."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM meetings ORDER BY date DESC, created_at DESC"
        ).fetchall()
    return json.dumps([dict(r) for r in rows], default=str)


# ── Summary ────────────────────────────────────────────────────────────────────

def get_db_summary() -> str:
    """Return counts and current property manager as JSON."""
    with get_conn() as conn:
        data = {
            "employees": conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0],
            "property_managers": conn.execute("SELECT COUNT(*) FROM property_managers").fetchone()[0],
            "key_contacts": conn.execute("SELECT COUNT(*) FROM key_contacts").fetchone()[0],
            "open_tasks": conn.execute("SELECT COUNT(*) FROM tasks WHERE status='open'").fetchone()[0],
            "high_priority_tasks": conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE priority='high' AND status='open'"
            ).fetchone()[0],
            "meetings": conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0],
        }
        pm = conn.execute(
            "SELECT name, email FROM property_managers WHERE status='current' LIMIT 1"
        ).fetchone()
        data["current_property_manager"] = dict(pm) if pm else None
    return json.dumps(data)
