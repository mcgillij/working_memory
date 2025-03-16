from pprint import pprint
from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import chromadb

from transformers import AutoTokenizer

EMBEDDING_DIR = "./content/chroma_db"
chroma_client = chromadb.PersistentClient(path=EMBEDDING_DIR)
MY_COLLECTION = "my_collection"

MODEL_NAME = "qwen2.5-14b-instruct"
MAX_TOKEN_LIMIT = 8000
MAX_VDB_RESULTS = 2
LLM_HOST = "http://192.168.2.35:1234/v1"


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
    client = OpenAI(base_url=LLM_HOST, api_key="dummy")
    response = client.chat.completions.create(
        model=model_name,
        messages=build_prompt(query, context),
    )
    pprint(response)
    return response.choices[0].message.content  # type: ignore


def main(chroma_collection) -> None:
    """main function to run the chatbot"""
    while True:
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue

        # Query the collection for relevant results
        results = chroma_collection.query(
            query_texts=[query],
            n_results=MAX_VDB_RESULTS,
            include=["documents", "metadatas"],
        )

        try:
            documents = [doc.strip() for doc in results["documents"][0]]

            if len(documents) > 1:
                # If multiple documents are returned, trim them to fit within the context limit
                trimmed_documents = []
                total_token_count = estimate_tokens(query)

                for document in documents:
                    document_tokens = estimate_tokens(document)
                    if total_token_count + document_tokens <= MAX_TOKEN_LIMIT - len(
                        query
                    ):
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
            print(
                f"The combined prompt exceeds the model's maximum capacity of {MAX_TOKEN_LIMIT} tokens. Trimming documents further."
            )

            trimmed_context = []
            total_token_count = estimate_tokens(query)

            for document in context:
                token_count = estimate_tokens(document)
                if total_token_count + token_count <= MAX_TOKEN_LIMIT - len(query):
                    trimmed_context.append(
                        trim_text(document, MAX_TOKEN_LIMIT - total_token_count)
                    )
                    total_token_count += token_count
                else:
                    break

            context = trimmed_context

        print(f"\nThinking using {MODEL_NAME}...\n")

        response = get_llm_response(query, context, MODEL_NAME)  # type: ignore

        pprint(response)
        print("\n")
        pprint(f"Source documents:\n{sources}")
        print("\n")


def get_tokenizer() -> AutoTokenizer:
    global tokenizer
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use a generic tokenizer
    return tokenizer


def estimate_tokens(text) -> int:
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def trim_text(text, max_length) -> str:
    while True:
        token_count = estimate_tokens(text)
        if token_count <= max_length:
            break
        words = text.split()
        text = " ".join(words[:-1])
    return text


if __name__ == "__main__":
    tokenizer = None
    get_tokenizer()
    chroma_collection = chroma_client.get_or_create_collection(name=MY_COLLECTION)
    main(chroma_collection)
