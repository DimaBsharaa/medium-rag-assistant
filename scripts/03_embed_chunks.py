import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

INPUT_PATH = Path("data/chunks_sample.csv")
OUTPUT_PATH = Path("data/chunks_sample_embedded.jsonl")

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
EXPECTED_DIMENSION = 1536
CHUNK_SIZE = 256


def require_env_var(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_embeddings_client():
    load_dotenv()

    api_key = require_env_var("OPENAI_API_KEY")
    base_url = require_env_var("OPENAI_BASE_URL")

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=base_url,
        api_key=api_key,
        chunk_size=CHUNK_SIZE,
    )


def clean_field(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def build_embedding_input(row):
    title = clean_field(row.get("title"))
    tags = clean_field(row.get("tags"))
    chunk_text = clean_field(row.get("chunk_text"))

    return f"Title: {title}\nTags: {tags}\nText: {chunk_text}"


def validate_embeddings(embeddings, expected_count):
    if len(embeddings) != expected_count:
        raise ValueError(
            f"Expected {expected_count} embeddings, but got {len(embeddings)}."
        )

    for index, embedding in enumerate(embeddings):
        if len(embedding) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Embedding at row {index} has dimension {len(embedding)}, "
                f"expected {EXPECTED_DIMENSION}."
            )


def write_jsonl(chunks_df, embeddings, output_path=OUTPUT_PATH):
    with output_path.open("w", encoding="utf-8") as file:
        for (_, row), embedding in zip(chunks_df.iterrows(), embeddings):
            item = {
                "id": f"{row['article_id']}_{row['chunk_index']}",
                "article_id": str(row["article_id"]),
                "title": row["title"],
                "authors": row["authors"],
                "url": row["url"],
                "timestamp": row["timestamp"],
                "tags": row["tags"],
                "chunk_index": int(row["chunk_index"]),
                "chunk_text": row["chunk_text"],
                "embedding": embedding,
            }
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def embed_chunks(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    if not input_path.exists():
        print(f"File not found: {input_path}")
        print("Run scripts/02_chunk_articles.py first.")
        return

    chunks_df = pd.read_csv(input_path)

    if "chunk_text" not in chunks_df.columns:
        raise ValueError(f"Missing required column in {input_path}: chunk_text")

    required_columns = ["title", "tags", "chunk_text"]
    for column in required_columns:
        if column not in chunks_df.columns:
            raise ValueError(f"Missing required column in {input_path}: {column}")

    texts = [build_embedding_input(row) for _, row in chunks_df.iterrows()]

    embeddings_client = build_embeddings_client()
    embeddings = embeddings_client.embed_documents(texts)

    validate_embeddings(embeddings, expected_count=len(chunks_df))
    write_jsonl(chunks_df, embeddings, output_path=output_path)

    print(f"Chunks embedded: {len(chunks_df)}")
    print(f"Embedding dimension: {EXPECTED_DIMENSION}")
    print(f"Output path: {output_path}")


def main():
    embed_chunks()


if __name__ == "__main__":
    main()
