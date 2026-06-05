from pathlib import Path
import importlib.util

INPUT_PATH = Path("data/chunks_full.csv")
OUTPUT_PATH = Path("data/chunks_full_embedded.jsonl")


def load_sample_embedder():
    script_path = Path(__file__).with_name("03_embed_chunks.py")
    spec = importlib.util.spec_from_file_location("embed_chunks", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    embedder = load_sample_embedder()
    embedder.embed_chunks(input_path=INPUT_PATH, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
