#!/usr/bin/env python
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Lista datasetów do przetworzenia",
    )
    parser.add_argument(
        "--chunking_types",
        type=str,
        nargs="+",
        required=True,
        help="Typ chunkingu: fixed-size lub recursive",
    )
    parser.add_argument(
        "--embed_chunks_types",
        type=str,
        nargs="+",
        required=True,
        help="",
    )
    parser.add_argument(
        "--vector_db",
        type=str,
        required=True,
        help="",
    )
    args = parser.parse_args()
    load_dotenv()

    SEARCH_DIR = Path(os.getenv('SEARCH_DIR'))
    VECTOR_DIR = Path(os.getenv('VECTOR_DIR'))
    DATASETS_DIR = Path(os.getenv('DATASETS_DIR'))
    
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        for chunking_type in args.chunking_types:
            for embed_chunk_type in args.embed_chunks_types:
                index_dir = VECTOR_DIR / dataset_name / chunking_type / embed_chunk_type / args.vector_db
                tasks_dir = DATASETS_DIR / dataset_name / 'Tasks'
                result_dir = SEARCH_DIR / dataset_name / chunking_type / embed_chunk_type / args.vector_db
                result_dir.mkdir(parents=True, exist_ok=True)
                # answer_questions(embed_chunk_type, args.vector_db, index_dir, result_dir)


if __name__ == "__main__":
    main()
