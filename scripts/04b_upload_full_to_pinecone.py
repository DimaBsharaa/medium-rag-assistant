from pathlib import Path
import importlib.util

INPUT_PATH = Path("data/chunks_full_embedded.jsonl")


def load_sample_uploader():
    script_path = Path(__file__).with_name("04_upload_to_pinecone.py")
    spec = importlib.util.spec_from_file_location("upload_to_pinecone", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    uploader = load_sample_uploader()
    uploader.upload_vectors(input_path=INPUT_PATH)


if __name__ == "__main__":
    main()
