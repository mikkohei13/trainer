import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "trainer.db"


def get_connection() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with get_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS project (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                taxon   TEXT NOT NULL UNIQUE,
                created TEXT NOT NULL
            )
        """)


def get_projects() -> list[sqlite3.Row]:
    with get_connection() as con:
        return con.execute("SELECT * FROM project ORDER BY taxon").fetchall()


def get_project(taxon: str) -> sqlite3.Row | None:
    with get_connection() as con:
        return con.execute(
            "SELECT * FROM project WHERE taxon = ?", (taxon,)
        ).fetchone()


def create_project(taxon: str) -> None:
    created = datetime.now(timezone.utc).isoformat()
    with get_connection() as con:
        con.execute(
            "INSERT INTO project (taxon, created) VALUES (?, ?)",
            (taxon, created),
        )
