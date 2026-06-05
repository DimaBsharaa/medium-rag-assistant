import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

INPUT_PATH = Path("data/chunks_sample_embedded.jsonl")
EXPECTED_DIMENSION = 1536
BATCH_SIZE = 25
UPSERT_TIMEOUT_SECONDS = 120


def require_env_var(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_vectors(input_path=INPUT_PATH):
    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path}. Run the embedding script first."
        )

    vectors = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            if len(row["embedding"]) != EXPECTED_DIMENSION:
                raise ValueError(
                    f"Line {line_number} has embedding dimension "
                    f"{len(row['embedding'])}, expected {EXPECTED_DIMENSION}."
                )

            vectors.append(
                {
                    "id": row["id"],
                    "values": row["embedding"],
                    "metadata": {
                        "article_id": str(row["article_id"]),
                        "title": row["title"],
                        "authors": row["authors"],
                        "url": row["url"],
                        "timestamp": row["timestamp"],
                        "tags": row["tags"],
                        "chunk_index": int(row["chunk_index"]),
                        "chunk_text": row["chunk_text"],
                    },
                }
            )

    return vectors


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def upload_vectors(input_path=INPUT_PATH):
    load_dotenv()

    api_key = require_env_var("PINECONE_API_KEY")
    index_name = require_env_var("PINECONE_INDEX_NAME")

    vectors = load_vectors(input_path=input_path)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    uploaded_count = 0
    for batch in iter_batches(vectors, BATCH_SIZE):
        index.upsert(vectors=batch, timeout=UPSERT_TIMEOUT_SECONDS)
        uploaded_count += len(batch)

    print(f"Vectors loaded from JSONL: {len(vectors)}")
    print(f"Vectors uploaded: {uploaded_count}")
    print(f"Pinecone index: {index_name}")


def main():
    upload_vectors()


if __name__ == "__main__":
    main()
