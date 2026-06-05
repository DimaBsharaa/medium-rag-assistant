import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/medium-english-50mb.csv")


def main():
    if not DATA_PATH.exists():
        print(f"File not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    print("Loaded CSV successfully!")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst article:")
    first = df.iloc[0]
    print("Title:", first.get("title"))
    print("Authors:", first.get("authors"))
    print("Tags:", first.get("tags"))
    print("Text preview:")
    print(str(first.get("text"))[:500])


if __name__ == "__main__":
    main()
