import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/medium-english-50mb.csv")
OUTPUT_PATH = Path("data/chunks_sample.csv")

ARTICLE_LIMIT = 100   # Use a small sample first to avoid wasting budget.

# The assignment reports chunk_size in tokens. This script uses a simple
# word-based approximation: 350 words is roughly 512 tokens for English text.
TARGET_CHUNK_SIZE_TOKENS = 512
CHUNK_SIZE_WORDS = 350
OVERLAP_RATIO = 0.1
OVERLAP_WORDS = 35


def clean_text(text):
    if pd.isna(text):  #handles missing text. If the CSV has NaN, we turn it into an empty string.
        return ""

    text = str(text)  #makes sure the value is a string
    text = re.sub(r"\s+", " ", text)  #replace repeated whitespace with one space
    return text.strip() #removes spaces from the beginning and end.

# This function takes one article text and splits it into chunks.
def chunk_text(text, chunk_size_words, overlap_words):
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0")

    if overlap_words < 0:
        raise ValueError("overlap_words must be 0 or greater")

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size_words - overlap_words

    for start in range(0, len(words), step):
        end = start + chunk_size_words
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def create_chunks(article_limit=ARTICLE_LIMIT, output_path=OUTPUT_PATH):
    if not DATA_PATH.exists():
        print(f"File not found: {DATA_PATH}")
        print("Make sure the CSV is inside the data folder.")
        return None

    df = pd.read_csv(DATA_PATH)
    articles_df = df.head(article_limit) if article_limit is not None else df

    rows = []

    for article_id, article in articles_df.iterrows():
        text = clean_text(article.get("text"))
        chunks = chunk_text(text, CHUNK_SIZE_WORDS, OVERLAP_WORDS)

        for chunk_index, chunk in enumerate(chunks):
            rows.append(
                {
                    "article_id": str(article_id),
                    "title": article.get("title"),
                    "authors": article.get("authors"),
                    "url": article.get("url"),
                    "timestamp": article.get("timestamp"),
                    "tags": article.get("tags"),
                    "chunk_index": chunk_index,
                    "chunk_text": chunk,
                }
            )

    chunks_df = pd.DataFrame(rows)
    chunks_df.to_csv(output_path, index=False)

    chunks_created = len(chunks_df)
    average_chunks = chunks_created / len(articles_df) if len(articles_df) else 0

    print(f"Input articles used: {len(articles_df)}")
    print(f"Chunks created: {chunks_created}")
    print(f"Average chunks per article: {average_chunks:.2f}")
    print(f"Output path: {output_path}")

    print("\nFirst 3 chunks:")
    for _, row in chunks_df.head(3).iterrows():
        print("\nTitle:", row["title"])
        print("Chunk index:", row["chunk_index"])
        print("Preview:", row["chunk_text"][:300])

    return chunks_df


def main():
    create_chunks()


if __name__ == "__main__":
    main()
