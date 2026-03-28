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
