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
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                taxon                   TEXT NOT NULL UNIQUE,
                created                 TEXT NOT NULL,
                active_training_run_id  INTEGER
            )
        """)
        _migrate_project_active_training_run(con)
        _migrate_project_active_quality_run(con)
        con.execute("""
            CREATE TABLE IF NOT EXISTS bounding_box (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path  TEXT NOT NULL,
                anno_id     TEXT NOT NULL,
                x           REAL NOT NULL,
                y           REAL NOT NULL,
                w           REAL NOT NULL,
                h           REAL NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS image_no_organism (
                image_path  TEXT PRIMARY KEY
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS training_run (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id   INTEGER NOT NULL REFERENCES project(id),
                status       TEXT NOT NULL,
                started_at   TEXT NOT NULL,
                finished_at  TEXT,
                model_path   TEXT,
                map50        REAL,
                map50_95     REAL,
                log_path     TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS quality_training_run (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id   INTEGER NOT NULL REFERENCES project(id),
                status       TEXT NOT NULL,
                started_at   TEXT NOT NULL,
                finished_at  TEXT,
                model_path   TEXT,
                val_rmse     REAL,
                log_path     TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS image_quality (
                image_path  TEXT PRIMARY KEY,
                quality     REAL NOT NULL
            )
        """)
        _migrate_image_quality(con)


def _migrate_project_active_training_run(con: sqlite3.Connection) -> None:
    cur = con.execute("PRAGMA table_info(project)")
    columns = {row[1] for row in cur.fetchall()}
    if "active_training_run_id" not in columns:
        con.execute(
            "ALTER TABLE project ADD COLUMN active_training_run_id INTEGER"
        )


def _migrate_project_active_quality_run(con: sqlite3.Connection) -> None:
    cur = con.execute("PRAGMA table_info(project)")
    columns = {row[1] for row in cur.fetchall()}
    if "active_quality_run_id" not in columns:
        con.execute(
            "ALTER TABLE project ADD COLUMN active_quality_run_id INTEGER"
        )


def _migrate_image_quality(con: sqlite3.Connection) -> None:
    cur = con.execute("PRAGMA table_info(image_quality)")
    columns = {row[1] for row in cur.fetchall()}
    if columns and "quality" not in columns:
        con.execute("ALTER TABLE image_quality ADD COLUMN quality REAL NOT NULL DEFAULT 0.0")


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


def save_annotations(
    image_path: str,
    boxes: list[dict],
    no_organism: bool,
) -> None:
    """
    Replace all stored annotation state for this image.
    Each box dict must have keys: anno_id, x, y, w, h.
    """
    with get_connection() as con:
        con.execute(
            "DELETE FROM bounding_box WHERE image_path = ?",
            (image_path,),
        )
        con.execute(
            "DELETE FROM image_no_organism WHERE image_path = ?",
            (image_path,),
        )
        if no_organism:
            con.execute(
                "INSERT INTO image_no_organism (image_path) VALUES (?)",
                (image_path,),
            )
        else:
            for box in boxes:
                con.execute(
                    """
                    INSERT INTO bounding_box
                    (image_path, anno_id, x, y, w, h)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        image_path,
                        box["anno_id"],
                        box["x"],
                        box["y"],
                        box["w"],
                        box["h"],
                    ),
                )


def get_annotations(image_path: str) -> dict:
    with get_connection() as con:
        no_row = con.execute(
            "SELECT 1 FROM image_no_organism WHERE image_path = ?",
            (image_path,),
        ).fetchone()
        no_organism = no_row is not None
        rows = con.execute(
            """
            SELECT anno_id, x, y, w, h
            FROM bounding_box
            WHERE image_path = ?
            ORDER BY id
            """,
            (image_path,),
        ).fetchall()
        boxes = []
        for row in rows:
            boxes.append({
                "anno_id": row["anno_id"],
                "x": row["x"],
                "y": row["y"],
                "w": row["w"],
                "h": row["h"],
            })
        return {"boxes": boxes, "no_organism": no_organism}


def save_image_quality(image_path: str, quality: float) -> None:
    with get_connection() as con:
        con.execute(
            """
            INSERT INTO image_quality (image_path, quality)
            VALUES (?, ?)
            ON CONFLICT(image_path) DO UPDATE SET quality = excluded.quality
            """,
            (image_path, quality),
        )


def delete_image_quality(image_path: str) -> None:
    with get_connection() as con:
        con.execute("DELETE FROM image_quality WHERE image_path = ?", (image_path,))


def get_image_quality_map(taxon: str) -> dict[str, float]:
    prefix = f"{taxon}/"
    like = prefix + "%"
    with get_connection() as con:
        rows = con.execute(
            "SELECT image_path, quality FROM image_quality WHERE image_path LIKE ?",
            (like,),
        ).fetchall()
    return {r["image_path"]: float(r["quality"]) for r in rows}


def create_training_run(project_id: int) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as con:
        cur = con.execute(
            "INSERT INTO training_run (project_id, status, started_at) VALUES (?, 'training', ?)",
            (project_id, started_at),
        )
        return cur.lastrowid


def create_quality_training_run(project_id: int) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as con:
        cur = con.execute(
            "INSERT INTO quality_training_run (project_id, status, started_at) VALUES (?, 'training', ?)",
            (project_id, started_at),
        )
        return cur.lastrowid


def finish_training_run(
    run_id: int,
    model_path: str,
    map50: float | None,
    map50_95: float | None,
    log_path: str,
) -> None:
    finished_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as con:
        con.execute(
            """
            UPDATE training_run
            SET status = 'done', finished_at = ?, model_path = ?,
                map50 = ?, map50_95 = ?, log_path = ?
            WHERE id = ?
            """,
            (finished_at, model_path, map50, map50_95, log_path, run_id),
        )


def finish_quality_training_run(
    run_id: int,
    model_path: str,
    val_rmse: float | None,
    log_path: str,
) -> None:
    finished_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as con:
        con.execute(
            """
            UPDATE quality_training_run
            SET status = 'done', finished_at = ?, model_path = ?,
                val_rmse = ?, log_path = ?
            WHERE id = ?
            """,
            (finished_at, model_path, val_rmse, log_path, run_id),
        )


def fail_training_run(run_id: int, log_path: str) -> None:
    finished_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as con:
        con.execute(
            "UPDATE training_run SET status = 'failed', finished_at = ?, log_path = ? WHERE id = ?",
            (finished_at, log_path, run_id),
        )


def fail_quality_training_run(run_id: int, log_path: str) -> None:
    finished_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as con:
        con.execute(
            "UPDATE quality_training_run SET status = 'failed', finished_at = ?, log_path = ? WHERE id = ?",
            (finished_at, log_path, run_id),
        )


def get_training_runs(project_id: int) -> list[sqlite3.Row]:
    with get_connection() as con:
        return con.execute(
            "SELECT * FROM training_run WHERE project_id = ? ORDER BY id DESC",
            (project_id,),
        ).fetchall()


def get_quality_training_runs(project_id: int) -> list[sqlite3.Row]:
    with get_connection() as con:
        return con.execute(
            "SELECT * FROM quality_training_run WHERE project_id = ? ORDER BY id DESC",
            (project_id,),
        ).fetchall()


def get_training_run(run_id: int) -> sqlite3.Row | None:
    with get_connection() as con:
        return con.execute(
            "SELECT * FROM training_run WHERE id = ?",
            (run_id,),
        ).fetchone()


def get_quality_training_run(run_id: int) -> sqlite3.Row | None:
    with get_connection() as con:
        return con.execute(
            "SELECT * FROM quality_training_run WHERE id = ?",
            (run_id,),
        ).fetchone()


def set_active_training_run(taxon: str, run_id: int | None) -> None:
    project = get_project(taxon)
    if project is None:
        raise ValueError("unknown project")

    if run_id is None:
        with get_connection() as con:
            con.execute(
                "UPDATE project SET active_training_run_id = NULL WHERE id = ?",
                (project["id"],),
            )
        return

    run = get_training_run(run_id)
    if run is None:
        raise ValueError("unknown training run")
    if run["project_id"] != project["id"]:
        raise ValueError("training run belongs to another project")
    if run["status"] != "done":
        raise ValueError("training run is not finished")
    if not run["model_path"]:
        raise ValueError("training run has no model file")

    with get_connection() as con:
        con.execute(
            "UPDATE project SET active_training_run_id = ? WHERE id = ?",
            (run_id, project["id"]),
        )


def set_active_quality_run(taxon: str, run_id: int | None) -> None:
    project = get_project(taxon)
    if project is None:
        raise ValueError("unknown project")

    if run_id is None:
        with get_connection() as con:
            con.execute(
                "UPDATE project SET active_quality_run_id = NULL WHERE id = ?",
                (project["id"],),
            )
        return

    run = get_quality_training_run(run_id)
    if run is None:
        raise ValueError("unknown quality training run")
    if run["project_id"] != project["id"]:
        raise ValueError("quality training run belongs to another project")
    if run["status"] != "done":
        raise ValueError("quality training run is not finished")
    if not run["model_path"]:
        raise ValueError("quality training run has no model file")

    with get_connection() as con:
        con.execute(
            "UPDATE project SET active_quality_run_id = ? WHERE id = ?",
            (run_id, project["id"]),
        )


def get_active_model_path_for_taxon(taxon: str) -> Path | None:
    """
    If the project has an active training run with status done and the weights
    file exists on disk, return that path. Otherwise None.
    """
    project = get_project(taxon)
    if project is None:
        return None
    active_id = project["active_training_run_id"]
    if active_id is None:
        return None
    run = get_training_run(int(active_id))
    if run is None:
        return None
    if run["status"] != "done":
        return None
    mp = run["model_path"]
    if not mp:
        return None
    path = Path(mp)
    if not path.is_file():
        return None
    return path


def get_active_quality_model_path_for_taxon(taxon: str) -> Path | None:
    project = get_project(taxon)
    if project is None:
        return None
    active_id = project["active_quality_run_id"]
    if active_id is None:
        return None
    run = get_quality_training_run(int(active_id))
    if run is None:
        return None
    if run["status"] != "done":
        return None
    mp = run["model_path"]
    if not mp:
        return None
    path = Path(mp)
    if not path.is_file():
        return None
    return path


def fail_stale_training_runs() -> None:
    """Mark any runs still in 'training' status as failed (app was restarted mid-run)."""
    finished_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as con:
        con.execute(
            "UPDATE training_run SET status = 'failed', finished_at = ? WHERE status = 'training'",
            (finished_at,),
        )


def fail_stale_quality_training_runs() -> None:
    finished_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as con:
        con.execute(
            "UPDATE quality_training_run SET status = 'failed', finished_at = ? WHERE status = 'training'",
            (finished_at,),
        )


def project_annotation_state(taxon: str) -> tuple[set[str], dict[str, int]]:
    """
    DB state for images under trainer/images/<taxon>/.
    Returns (paths marked no-organism, image_path -> bounding box count).
    """
    prefix = f"{taxon}/"
    like = prefix + "%"
    with get_connection() as con:
        no_rows = con.execute(
            "SELECT image_path FROM image_no_organism WHERE image_path LIKE ?",
            (like,),
        ).fetchall()
        no_set = {r["image_path"] for r in no_rows}
        box_rows = con.execute(
            """
            SELECT image_path, COUNT(*) AS n
            FROM bounding_box
            WHERE image_path LIKE ?
            GROUP BY image_path
            """,
            (like,),
        ).fetchall()
        box_map = {r["image_path"]: int(r["n"]) for r in box_rows}
    return no_set, box_map
