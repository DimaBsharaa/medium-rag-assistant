import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
TOP_K = 7
QUESTION = (
    "Find an article about writing better headlines. "
    "Provide the title and explain why it is relevant."
)


def require_env_var(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_embeddings_client():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def main():
    load_dotenv()

    require_env_var("OPENAI_API_KEY")
    require_env_var("OPENAI_BASE_URL")
    pinecone_api_key = require_env_var("PINECONE_API_KEY")
    index_name = require_env_var("PINECONE_INDEX_NAME")

    embeddings_client = build_embeddings_client()
    query_vector = embeddings_client.embed_query(QUESTION)

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
    )

    print("Question:", QUESTION)
    print("Top K:", TOP_K)

    for rank, match in enumerate(results.matches, start=1):
        metadata = match.metadata or {}
        chunk_text = metadata.get("chunk_text", "")

        print(f"\nRank: {rank}")
        print(f"Score: {match.score}")
        print(f"Article ID: {metadata.get('article_id')}")
        print(f"Chunk index: {metadata.get('chunk_index')}")
        print(f"Title: {metadata.get('title')}")
        print(f"Chunk preview: {chunk_text[:300]}")


if __name__ == "__main__":
    main()
