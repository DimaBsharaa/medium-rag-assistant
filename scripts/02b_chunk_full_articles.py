from pathlib import Path
import importlib.util

OUTPUT_PATH = Path("data/chunks_full.csv")


def load_sample_chunker():
    script_path = Path(__file__).with_name("02_chunk_articles.py")
    spec = importlib.util.spec_from_file_location("chunk_articles", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    chunker = load_sample_chunker()
    chunker.create_chunks(article_limit=None, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
