import json
from collections import namedtuple
from datetime import datetime
from pprint import pprint
from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import primp
import sqlite3
from datetime import datetime
import chromadb

from transformers import AutoTokenizer

embedding_directory = "./content/chroma_db"
chroma_client = chromadb.PersistentClient(path=embedding_directory)
my_collection = "my_collection"

model_to_use = "qwen2.5-14b-instruct"
MAX_TOKEN_LIMIT = 8000
MAX_VDB_RESULTS = 2
llm_hostname = "http://192.168.2.35:1234/v1"

BookmarkEntry = namedtuple('BookmarkEntry', ['uri', 'title', 'last_modified'])


def convert_timestamp_to_datetime(timestamp_us):
    """Convert a timestamp in microseconds to a datetime object."""
    return datetime.fromtimestamp(timestamp_us / 1_000_000.0)

# Function to extract and store all links from the bookmarks data
def extract_bookmarks(bookmark_node, bookmarks_list):
    if isinstance(bookmark_node, dict):
        if bookmark_node.get('type') == "text/x-moz-place":
            # This is a bookmark entry with a URL
            uri = bookmark_node.get('uri')
            if uri and uri.startswith('http'):
                title = bookmark_node.get('title')
                last_modified_us = bookmark_node.get('lastModified')
                last_modified_dt = convert_timestamp_to_datetime(last_modified_us)
                bookmarks_list.append(BookmarkEntry(uri=uri, title=title, last_modified=last_modified_dt))
        elif 'children' in bookmark_node:
            # This is a folder containing other bookmarks or folders
            for child in bookmark_node['children']:
                extract_bookmarks(child, bookmarks_list)

# Function to convert datetime objects to a string format for SQLite
def adapt_datetime(dt):
    return dt.isoformat()

# Register the adapter with sqlite3
sqlite3.register_adapter(datetime, adapt_datetime)

def insert_or_update_bookmark(bookmark, content=None, checked=False):
    if not bookmark.uri or not bookmark.title or not bookmark.last_modified:
        print(bookmark)
        raise ValueError("Bookmark URI, title, and last_modified must be provided.")

    # Check if the bookmark already exists in the database
    cursor.execute('SELECT * FROM bookmarks WHERE uri = ?', (bookmark.uri,))
    existing_bookmark = cursor.fetchone()

    if existing_bookmark and checked:
        # do noting
        cursor.execute('''
            UPDATE bookmarks 
            SET title = ?, last_modified = ?, content = ?, checked = ?
            WHERE uri = ?
            ''', (bookmark.title, bookmark.last_modified, content, checked, bookmark.uri))

    elif existing_bookmark and not checked:
        pass
    else:
        # Insert a new bookmark
        print("New value found inserting")
        cursor.execute('''
        INSERT INTO bookmarks (uri, title, last_modified, content, checked)
        VALUES (?, ?, ?, ?, ?)
        ''', (bookmark.uri, bookmark.title, bookmark.last_modified, content, checked))

    conn.commit()

def createdb():
        # Create the table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookmarks (
            uri TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            last_modified TIMESTAMP NOT NULL,
            content TEXT,
            checked BOOLEAN
        )
        ''')

# Function to fetch text content from a URI using primp
def fetch_text_content(uri):
    if uri and uri.startswith('http'):
        client = primp.Client(impersonate="firefox_135", impersonate_os="linux")
        resp = client.get(uri)
        return resp.text_plain

# Function to update the database with fetched text content and set checked to True
def update_bookmarks_with_text_content():
    # Query for bookmarks where checked is False
    cursor.execute('SELECT uri, title, last_modified FROM bookmarks WHERE checked = 0')
    rows = cursor.fetchall()

    for row in rows:
        uri, title, last_modified = row
        bookmark = BookmarkEntry(uri=uri, title=title, last_modified=last_modified)

        try:
            # Fetch text content
            text_content = fetch_text_content(uri)

            # Update the database with the fetched text content and set checked to True
            insert_or_update_bookmark(bookmark, content=text_content, checked=True)
            print(f"Updated bookmark for URI: {uri}")

        except Exception as e:
            print(f"Failed to fetch or update bookmark for URI: {uri}. Error: {e}")

def get_content():
    cursor.execute('SELECT title, uri, content FROM bookmarks WHERE checked = 1')
    rows = cursor.fetchall()
    return rows


def build_prompt(query: str, context: List[str]) -> List[ChatCompletionMessageParam]:
    system: ChatCompletionMessageParam = {
        "role": "system",
        "content": "I am going to ask you a question, which I would like you to answer"
        "based only on the provided context, and not any other information."
        "If there is not enough information in the context to answer the question,"
        'say "I am not sure", then try to make a guess.'
        "Break your answer up into nicely readable paragraphs.",
    }
    user: ChatCompletionMessageParam = {
        "role": "user",
        "content": f"The question is {query}. Here is all the context you have:"
        f'{(" ").join(context)}',
    }

    return [system, user]


def get_llm_response(query: str, context: List[str], model_name: str) -> str:
    """
    Queries the LLM API to get a response to the question.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """
    client = OpenAI(base_url=llm_hostname, api_key='dummy')
    response = client.chat.completions.create(
        model=model_name,
        messages=build_prompt(query, context),
    )
    pprint(response)
    return response.choices[0].message.content  # type: ignore


def main(chroma_collection) -> None:
    model_name = model_to_use

    while True:
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue

        # Query the collection for relevant results
        results = chroma_collection.query(
            query_texts=[query], n_results=MAX_VDB_RESULTS, include=["documents", "metadatas"]
        )

        try:
            documents = [doc.strip() for doc in results["documents"][0]]

            if len(documents) > 1:
                # If multiple documents are returned, trim them to fit within the context limit
                trimmed_documents = []
                total_token_count = estimate_tokens(query)

                for document in documents:
                    document_tokens = estimate_tokens(document)
                    if total_token_count + document_tokens <= MAX_TOKEN_LIMIT - len(query):
                        trimmed_documents.append(document)
                        total_token_count += document_tokens
                    else:
                        break

                documents = trimmed_documents

            print(f"Found {len(documents)} documents.")

            # Access metadatas and documents separately
            metadatas = results["metadatas"][0]
            sources = "\n".join(
                [
                    f"uri: {metadata['source']}: title: {metadata['title']}"
                    for metadata in metadatas  # type: ignore
                ]
            )
        except (KeyError, TypeError) as e:
            print(f"An error occurred while processing the results: {e}")
            print("Please check the structure of the 'results' dictionary.")

        context = documents

        if not context:
            print("No relevant documents found.")
            continue

        # Check token count before sending to LLM
        query_plus_documents = query + " ".join(context)

        if estimate_tokens(query_plus_documents) > MAX_TOKEN_LIMIT:
            print(f"The combined prompt exceeds the model's maximum capacity of {MAX_TOKEN_LIMIT} tokens. Trimming documents further.")

            trimmed_context = []
            total_token_count = estimate_tokens(query)

            for document in context:
                token_count = estimate_tokens(document)
                if total_token_count + token_count <= MAX_TOKEN_LIMIT - len(query):
                    trimmed_context.append(trim_text(document, MAX_TOKEN_LIMIT - total_token_count))
                    total_token_count += token_count
                else:
                    break

            context = trimmed_context

        print(f"\nThinking using {model_name}...\n")

        response = get_llm_response(query, context, model_name)  # type: ignore

        pprint(response)
        print("\n")
        pprint(f"Source documents:\n{sources}")
        print("\n")


# def main(chroma_collection) -> None:
    # model_name = model_to_use
    # while True:
        # query = input("Query: ")
        # if len(query) == 0:
            # print("Please enter a question. Ctrl+C to Quit.\n")
            # continue

        # # Query the collection to get the 5 most relevant results
        # results = chroma_collection.query(
            # query_texts=[query], n_results=MAX_VDB_RESULTS, include=["documents", "metadatas"]
        # )

        # try:
            # documents = [doc.strip() for doc in results["documents"][0]]
            # print(f"Found {len(documents)} documents.")
            # # check if the documents are too long
            # query_plus_documents = query + " ".join(documents)
            # encoding = tokenizer.encode(query_plus_documents, truncation=True, max_length=MAX_TOKEN_LIMIT)
            # if len(encoding) > MAX_TOKEN_LIMIT:
                # print("The combined query and documents are too long. Trimming the documents.")
                # documents = [tokenizer.decode(tokenizer.encode(doc, truncation=True, max_length=MAX_TOKEN_LIMIT//MAX_VDB_RESULTS - estimate_tokens(query)))[:MAX_TOKEN_LIMIT//2] for doc in documents]
            # # if estimate_tokens(query_plus_documents) > MAX_TOKEN_LIMIT:

                # # print("The combined query and documents are too long. Trimming the documents.")
                # # documents = [trim_text(doc, MAX_TOKEN_LIMIT//MAX_VDB_RESULTS - estimate_tokens(query)) for doc in documents]

            # # Access metadatas and documents separately
            # metadatas = results["metadatas"][0]
            # sources = "\n".join(
                # [
                    # f"uri: {metadata['source']}: title: {metadata['title']}"
                    # for metadata in metadatas  # type: ignore
                # ]
            # )
        # except (KeyError, TypeError) as e:
            # print(f"An error occurred while processing the results: {e}")
            # print("Please check the structure of the 'results' dictionary.")

        # print(f"\nThinking using {model_name}...\n")
        # response = get_llm_response(query, documents, model_name)  # type: ignore

        # # Output, with sources
        # pprint(response)
        # print("\n")
        # pprint(f"Source documents:\n{sources}")
        # print("\n")

def seed_chroma(content):
    content_list = []
    content_metadata = []
    # get content ready for tossing into chromadb
    for i in content:
        title, uri, c = i
        content_list.append(c)
        content_metadata.append({"source": uri, "title": title})

    id_array = [f'id{i+1}' for i in range(len(content_list))]

    chroma_collection = chroma_client.get_or_create_collection(name=my_collection)
    chroma_collection.upsert(
        documents=content_list,
        metadatas=content_metadata,
        ids=id_array
    )
    return chroma_collection

def parse_bookmarks():
    # Define the BookmarkEntry named tuple
    bookmarks_file = "bookmarks-2025-03-15.json"
    # Initialize the list to store the bookmarks
    bookmarks = []

    with open(bookmarks_file, 'r', encoding='utf-8') as f:
        file_data = json.load(f)

    # Start extracting bookmarks from the root of the bookmarks data
    extract_bookmarks(file_data, bookmarks)
    return bookmarks

def process_bookmarks():
    bookmarks = parse_bookmarks()
    content = []
    # Insert all extracted bookmarks into the database
    try:
        conn = sqlite3.connect('bookmarks.db')
        cursor = conn.cursor()
        createdb()
        for entry in bookmarks:
            insert_or_update_bookmark(entry)
        content = get_content()
        update_bookmarks_with_text_content()
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

def get_tokenizer():
    global tokenizer
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use a generic tokenizer
    return tokenizer

def estimate_tokens(text):
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def trim_text(text, max_length):
    while True:
        token_count = estimate_tokens(text)
        if token_count <= max_length:
            break
        words = text.split()
        text = ' '.join(words[:-1])
    return text


if __name__ == "__main__":
    tokenizer = None
    get_tokenizer()
    seed_chromadb = False

    if seed_chromadb:
        content = process_bookmarks()
        chroma_collection = seed_chroma(content) # uncomment to seed
    else:
        chroma_collection = chroma_client.get_or_create_collection(name=my_collection)

    main(chroma_collection)
