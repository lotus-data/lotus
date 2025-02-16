import sqlite3

import lotus
from lotus.data_connectors import DataConnector
from lotus.models import LM

conn = sqlite3.connect("example_movies.db")
cursor = conn.cursor()

# Create the table
cursor.execute("""
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,
    title TEXT,
    director TEXT,
    rating REAL,
    release_year INTEGER,
    description TEXT
)
""")

cursor.execute("DELETE FROM movies")

# Insert sample data
cursor.executemany(
    """
INSERT INTO movies (title, director, rating, release_year, description)
VALUES (?, ?, ?, ?, ?)
""",
    [
        ("The Matrix", "Wachowskis", 8.7, 1999, "A hacker discovers the reality is simulated."),
        ("The Godfather", "Francis Coppola", 9.2, 1972, "The rise and fall of a powerful mafia family."),
        ("Inception", "Christopher Nolan", 8.8, 2010, "A thief enters dreams to steal secrets."),
        ("Parasite", "Bong Joon-ho", 8.6, 2019, "A poor family schemes to infiltrate a rich household."),
        ("Interstellar", "Christopher Nolan", 8.6, 2014, "A team travels through a wormhole to save humanity."),
        ("Titanic", "James Cameron", 7.8, 1997, "A love story set during the Titanic tragedy."),
    ],
)

conn.commit()
conn.close()


lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

query = "SELECT * FROM movies"
df = DataConnector.load_from_db("sqlite:///example_movies.db", query=query)

df = df.sem_filter("the {title} is science fiction")
print(df)
