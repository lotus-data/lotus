from __future__ import annotations

import sqlite3
from pathlib import Path

import lotus
from lotus.data_connectors import DataConnector
from lotus.models import LM

# Always place DB next to this file (stable, no “random working dir” DBs)
DB_PATH = Path(__file__).resolve().with_name("example_papers.db")

# Choose behavior:
# - "insert_missing": only insert rows whose id doesn't exist (fast, no changes to existing)
# - "upsert": insert new rows and update title/abstract for existing ids
SEED_MODE = "insert_missing"


ROWS = [
    (
        100,
        "Quantum Networks",
        "This paper explores quantum entanglement to build distributed communication networks with improved security.",
    ),
    (
        101,
        "AI Ethics",
        "We discuss fairness, accountability, and transparency challenges in autonomous AI systems deployed at scale.",
    ),
    (
        102,
        "Climate Modeling",
        "This study models long-term climate prediction using deep learning simulation techniques and uncertainty estimation.",
    ),
    (
        103,
        "Database Optimization",
        "We propose indexing strategies and query rewriting methods to reduce latency for analytical workloads.",
    ),
]


def seed(db_path: Path, rows: list[tuple[int, str, str]], mode: str = "insert_missing") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL
        )
        """
    )

    if mode == "insert_missing":
        # Only insert new ids; do not overwrite existing rows
        cur.executemany(
            """
            INSERT OR IGNORE INTO papers (id, title, abstract)
            VALUES (?, ?, ?)
            """,
            rows,
        )
    elif mode == "upsert":
        # Insert new ids; update existing ids
        cur.executemany(
            """
            INSERT INTO papers (id, title, abstract)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                abstract = excluded.abstract
            """,
            rows,
        )
    else:
        conn.close()
        raise ValueError("mode must be 'insert_missing' or 'upsert'")

    conn.commit()
    conn.close()


def main() -> None:
    seed(DB_PATH, ROWS, mode=SEED_MODE)

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    df = DataConnector.load_from_db(
        f"sqlite:///{DB_PATH}",
        query="SELECT id, title, abstract FROM papers",
    )

    # make provenance equal DB primary key
    df = df.set_index("id")

    out = df.sem_map(
        "Summarize {abstract} in one concise sentence.",
        return_provenance=True,
        provenance_col="paper_id",
        track_pipeline=True,
        op_name="sql_sem_map",
        progress_bar_desc="SQL SemMap",
    )

    print(f"\nDB location: {DB_PATH}")
    print("\n=== SEM_MAP OUTPUT ===")
    print(out)

    print("\n=== PIPELINE PROVENANCE (attrs['_prov']) ===")
    print(out.attrs.get("_prov", []))


if __name__ == "__main__":
    main()
