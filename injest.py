import json
from collections import namedtuple

import primp
import sqlite3
from datetime import datetime
import chromadb
from typing import List, Dict

BOOKMARKS_FILE = "bookmarks-2025-03-15.json"

EMBEDDING_DIR = "./content/chroma_db"
chroma_client = chromadb.PersistentClient(path=EMBEDDING_DIR)
MY_COLLECTION = "my_collection"

BookmarkEntry = namedtuple("BookmarkEntry", ["uri", "title", "last_modified"])


def seed_chroma(content) -> chromadb.Collection:
    if len(content) == 0:
        raise ValueError("Content list passed to seed_chroma must be non-empty")

    content_list = []
    content_metadata = []
    for i in content:
        title, uri, c = i
        content_list.append(c)
        content_metadata.append({"source": uri, "title": title})

    id_array = [f"id{i+1}" for i in range(len(content_list))]

    chroma_collection = chroma_client.get_or_create_collection(name=MY_COLLECTION)

    if not content_list or not content_metadata or not id_array:
        raise ValueError("Empty list found for documents, metadatas, or ids")

    chroma_collection.upsert(
        documents=content_list, metadatas=content_metadata, ids=id_array
    )
    return chroma_collection


def parse_bookmarks(file_to_parse) -> List:
    bookmarks = []

    with open(file_to_parse, "r", encoding="utf-8") as f:
        file_data = json.load(f)

    extract_bookmarks(file_data, bookmarks)
    return bookmarks


def process_bookmarks() -> List:
    bookmarks = parse_bookmarks(BOOKMARKS_FILE)
    print(f"Number of bookmarks extracted: {len(bookmarks)}")
    content = []

    # Insert all extracted bookmarks into the database and fetch their content.
    try:
        conn = sqlite3.connect("bookmarks.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bookmarks (
                uri TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                last_modified TIMESTAMP NOT NULL,
                content TEXT,
                checked BOOLEAN
            )
        """
        )
        for entry in bookmarks:
            insert_or_update_bookmark(
                cursor, entry, None, False
            )  # Assuming you want to insert without setting checked=True initially

        conn.commit()

        # Fetch content from the database.
        updated_list = update_content(cursor)
        print(f"Updated: {len(updated_list)} bookmarks")
        conn.commit()

        content = fetch_updated_records(cursor)
        print(f"Content: {len(content)}")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

    finally:
        if conn:
            conn.close()
    return content


def convert_timestamp_to_datetime(timestamp_us: int) -> datetime:
    """Convert a timestamp in microseconds to a datetime object."""
    return datetime.fromtimestamp(timestamp_us / 1_000_000.0)


# Function to extract and store all links from the bookmarks data
def extract_bookmarks(bookmark_node: Dict, bookmarks_list: List) -> None:
    if isinstance(bookmark_node, dict):
        if bookmark_node.get("type") == "text/x-moz-place":
            # This is a bookmark entry with a URL
            uri = bookmark_node.get("uri")
            if uri and uri.startswith("http"):
                title = bookmark_node.get("title")
                last_modified_us = bookmark_node.get("lastModified")
                last_modified_dt = convert_timestamp_to_datetime(last_modified_us)
                bookmarks_list.append(
                    BookmarkEntry(uri=uri, title=title, last_modified=last_modified_dt)
                )
        elif "children" in bookmark_node:
            # This is a folder containing other bookmarks or folders
            for child in bookmark_node["children"]:
                extract_bookmarks(child, bookmarks_list)


# Function to convert datetime objects to a string format for SQLite
def adapt_datetime(dt: datetime) -> str:
    return dt.isoformat()


def insert_or_update_bookmark(
    cursor: sqlite3.Cursor, bookmark: BookmarkEntry, content=None, checked: bool = False
):
    if not bookmark.uri or not bookmark.title or not bookmark.last_modified:
        print(bookmark)
        raise ValueError("Bookmark URI, title, and last_modified must be provided.")

    # Check if the bookmark already exists in the database
    cursor.execute("SELECT * FROM bookmarks WHERE uri = ?", (bookmark.uri,))
    existing_bookmark = cursor.fetchone()

    if existing_bookmark and checked:
        # do noting
        cursor.execute(
            """
            UPDATE bookmarks 
            SET title = ?, last_modified = ?, content = ?, checked = ?
            WHERE uri = ?
            """,
            (bookmark.title, bookmark.last_modified, content, checked, bookmark.uri),
        )

    elif existing_bookmark and not checked:
        pass
    else:
        # Insert a new bookmark
        print("New value found inserting")
        cursor.execute(
            """
        INSERT INTO bookmarks (uri, title, last_modified, content, checked)
        VALUES (?, ?, ?, ?, ?)
        """,
            (bookmark.uri, bookmark.title, bookmark.last_modified, content, checked),
        )


# Function to fetch text content from a URI using primp
def fetch_text_content(uri: str) -> str:
    if uri and uri.startswith("http"):
        client = primp.Client(impersonate="firefox_135", impersonate_os="linux")
        resp = client.get(uri)
        return resp.text_plain


def update_content(cursor: sqlite3.Cursor) -> List:
    # Fetch text content for all bookmarks and update them in the database.
    cursor.execute("SELECT uri, title, last_modified FROM bookmarks WHERE checked = 0")

    updated_bookmarks = []
    rows = cursor.fetchall()

    for row in rows:
        uri, title, last_modified = row
        bookmark_entry = BookmarkEntry(
            uri=uri, title=title, last_modified=last_modified
        )

        try:
            text_content = fetch_text_content(uri)
            insert_or_update_bookmark(
                cursor, bookmark_entry, content=text_content, checked=True
            )

            updated_bookmarks.append(
                (bookmark_entry.title, bookmark_entry.uri, text_content)
            )
        except Exception as e:
            print(f"Failed to fetch or update bookmark for URI: {uri}. Error: {e}")

    return updated_bookmarks


def fetch_updated_records(cursor: sqlite3.Cursor) -> List:
    # Fetch text content for all bookmarks and update them in the database.
    cursor.execute("SELECT uri, title, content FROM bookmarks WHERE checked = 1")

    content = []
    rows = cursor.fetchall()

    for row in rows:
        uri, title, c = row
        content.append((title, uri, c))

    return content


if __name__ == "__main__":
    # Register the adapter with sqlite3
    sqlite3.register_adapter(datetime, adapt_datetime)
    content = process_bookmarks()
    chroma_collection = seed_chroma(content)
