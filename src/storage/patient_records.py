"""
Patient records storage.

Stores patients and their repeated health measurements so that a
single patient can be followed over time instead of being assessed
only once.

Storage backend: SQLite (standard library, no extra dependency).
Database file: <project_root>/data/patient_records.db
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


# ============================================================
# 1. DATABASE LOCATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "patient_records.db"


# Measurement columns exposed to the UI, in display order.
MEASUREMENT_FIELDS = [
    "measured_at",
    "age",
    "height",
    "weight",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "glucose",
    "total_cholesterol",
    "cardio_probability",
    "diabetes_probability",
    "hypertension_probability",
    "overall_risk",
    "priority_disease",
    "note",
]


# ============================================================
# 2. CONNECTION
# ============================================================

@contextmanager
def _connect():
    """Open a SQLite connection with row access by column name."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db():
    """Create the tables if they do not exist yet."""

    with _connect() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                gender TEXT,
                birth_year INTEGER,
                note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                measured_at TEXT NOT NULL,
                age INTEGER,
                height REAL,
                weight REAL,
                bmi REAL,
                systolic_bp REAL,
                diastolic_bp REAL,
                heart_rate REAL,
                glucose REAL,
                total_cholesterol REAL,
                cardio_probability REAL,
                diabetes_probability REAL,
                hypertension_probability REAL,
                overall_risk TEXT,
                priority_disease TEXT,
                note TEXT,
                FOREIGN KEY (patient_id)
                    REFERENCES patients (id)
                    ON DELETE CASCADE
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_measurements_patient
            ON measurements (patient_id, measured_at)
            """
        )


# ============================================================
# 3. PATIENTS
# ============================================================

def add_patient(name, gender=None, birth_year=None, note=None):
    """Create a patient and return its id.

    Raises ValueError when the name is empty or already used.
    """

    clean_name = (name or "").strip()

    if not clean_name:
        raise ValueError("Patient name is required.")

    init_db()

    with _connect() as connection:

        existing = connection.execute(
            "SELECT id FROM patients WHERE name = ?",
            (clean_name,)
        ).fetchone()

        if existing is not None:
            raise ValueError(
                f"Patient '{clean_name}' already exists."
            )

        cursor = connection.execute(
            """
            INSERT INTO patients
                (name, gender, birth_year, note, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                clean_name,
                gender,
                birth_year,
                note,
                datetime.now().isoformat(timespec="seconds"),
            )
        )

        return cursor.lastrowid


def list_patients():
    """Return all patients with their measurement statistics."""

    init_db()

    with _connect() as connection:

        rows = connection.execute(
            """
            SELECT
                p.id,
                p.name,
                p.gender,
                p.birth_year,
                p.note,
                p.created_at,
                COUNT(m.id) AS measurement_count,
                MAX(m.measured_at) AS last_measured_at
            FROM patients p
            LEFT JOIN measurements m
                ON m.patient_id = p.id
            GROUP BY p.id
            ORDER BY p.name COLLATE NOCASE
            """
        ).fetchall()

        return [dict(row) for row in rows]


def get_patient(patient_id):
    """Return one patient as a dict, or None when not found."""

    init_db()

    with _connect() as connection:

        row = connection.execute(
            "SELECT * FROM patients WHERE id = ?",
            (patient_id,)
        ).fetchone()

        return dict(row) if row is not None else None


def delete_patient(patient_id):
    """Delete a patient together with all of their measurements."""

    init_db()

    with _connect() as connection:
        connection.execute(
            "DELETE FROM measurements WHERE patient_id = ?",
            (patient_id,)
        )
        connection.execute(
            "DELETE FROM patients WHERE id = ?",
            (patient_id,)
        )


# ============================================================
# 4. MEASUREMENTS
# ============================================================

def add_measurement(patient_id, values, measured_at=None):
    """Store one measurement for a patient and return its id.

    `values` is a dict that may contain any key of
    MEASUREMENT_FIELDS except `measured_at`; missing keys are
    stored as NULL.
    """

    init_db()

    if get_patient(patient_id) is None:
        raise ValueError("Patient does not exist.")

    columns = [
        field for field in MEASUREMENT_FIELDS
        if field != "measured_at"
    ]

    timestamp = (
        measured_at
        or datetime.now().isoformat(timespec="seconds")
    )

    placeholders = ", ".join(["?"] * (len(columns) + 2))

    sql = (
        "INSERT INTO measurements "
        f"(patient_id, measured_at, {', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )

    parameters = [patient_id, timestamp] + [
        values.get(column) for column in columns
    ]

    with _connect() as connection:
        cursor = connection.execute(sql, parameters)
        return cursor.lastrowid


def list_measurements(patient_id, limit=None):
    """Return the measurements of a patient, oldest first."""

    init_db()

    sql = (
        "SELECT id, " + ", ".join(MEASUREMENT_FIELDS) + " "
        "FROM measurements WHERE patient_id = ? "
        "ORDER BY measured_at ASC, id ASC"
    )

    parameters = [patient_id]

    if limit is not None:
        sql += " LIMIT ?"
        parameters.append(limit)

    with _connect() as connection:
        rows = connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]


def delete_measurement(measurement_id):
    """Delete a single measurement."""

    init_db()

    with _connect() as connection:
        connection.execute(
            "DELETE FROM measurements WHERE id = ?",
            (measurement_id,)
        )


def count_measurements(patient_id):
    """Return how many measurements a patient has."""

    init_db()

    with _connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total FROM measurements "
            "WHERE patient_id = ?",
            (patient_id,)
        ).fetchone()

        return int(row["total"])
