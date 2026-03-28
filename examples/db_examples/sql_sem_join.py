from __future__ import annotations

import sqlite3
from pathlib import Path

import lotus
from lotus.data_connectors import DataConnector
from lotus.models import LM

# Always place DB next to this file (stable, no “random working dir” DBs)
DB_PATH = Path(__file__).resolve().with_name("example_movies.db")

# Choose behavior:
# - "insert_missing": only insert rows whose id doesn't exist (no overwrites)
# - "upsert": insert new rows and update existing rows if ids match
SEED_MODE = "insert_missing"

MOVIES_ROWS = [
    (0, "The Matrix", "Wachowskis", 8.7, 1999, "A hacker discovers the reality is simulated."),
    (1, "The Godfather", "Francis Coppola", 9.2, 1972, "The rise and fall of a powerful mafia family."),
    (2, "Inception", "Christopher Nolan", 8.8, 2010, "A thief enters dreams to steal secrets."),
    (3, "Parasite", "Bong Joon-ho", 8.6, 2019, "A poor family schemes to infiltrate a rich household."),
    (4, "Interstellar", "Christopher Nolan", 8.6, 2014, "A team travels through a wormhole to save humanity."),
    (5, "Titanic", "James Cameron", 7.8, 1997, "A love story set during the Titanic tragedy."),
]

CATEGORIES_ROWS = [
    (10, "Science Fiction"),
    (11, "Crime / Mafia"),
    (12, "Thriller / Heist"),
    (13, "Drama"),
    (14, "Romance"),
]


def seed(db_path: Path, mode: str = "insert_missing") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Tables
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            director TEXT NOT NULL,
            rating REAL NOT NULL,
            release_year INTEGER NOT NULL,
            description TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL
        )
        """
    )

    if mode == "insert_missing":
        # Insert only new rows; ignore if id already exists
        cur.executemany(
            """
            INSERT OR IGNORE INTO movies (id, title, director, rating, release_year, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            MOVIES_ROWS,
        )
        cur.executemany(
            """
            INSERT OR IGNORE INTO categories (id, category)
            VALUES (?, ?)
            """,
            CATEGORIES_ROWS,
        )

    elif mode == "upsert":
        # Insert new rows; update existing rows on id collision
        cur.executemany(
            """
            INSERT INTO movies (id, title, director, rating, release_year, description)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                director = excluded.director,
                rating = excluded.rating,
                release_year = excluded.release_year,
                description = excluded.description
            """,
            MOVIES_ROWS,
        )
        cur.executemany(
            """
            INSERT INTO categories (id, category)
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET
                category = excluded.category
            """,
            CATEGORIES_ROWS,
        )
    else:
        conn.close()
        raise ValueError("mode must be 'insert_missing' or 'upsert'")

    conn.commit()
    conn.close()


def main() -> None:
    seed(DB_PATH, mode=SEED_MODE)

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    movies = DataConnector.load_from_db(
        f"sqlite:///{DB_PATH}",
        query="SELECT id, title, description FROM movies",
    )
    cats = DataConnector.load_from_db(
        f"sqlite:///{DB_PATH}",
        query="SELECT id, category FROM categories",
    )

    # provenance == DB primary keys
    movies = movies.set_index("id")
    cats = cats.set_index("id")

    out = movies.sem_join(
        cats,
        "the {title} belongs to the {category}.",
        return_provenance=True,
        provenance_left_col="movie_id",
        provenance_right_col="category_id",
    )

    print(f"\nDB location: {DB_PATH}")
    print("\n=== SEM_JOIN OUTPUT ===")
    print(out)


if __name__ == "__main__":
    main()
