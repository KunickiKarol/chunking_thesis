from src.tools.logging_config import setup_logging

setup_logging()

# from src.download_datasets.download_datasets_all import main
# main()

# from src.preprocess_datasets.preprocess_datasets_all import main
# main()

from src.chunking.chunking_text_all import main
main()

# from src.embed_chunks.embed_chunks_all import main
# main()

# from src.search.search_all import main
# main()

# from src.rerank.rerank_all import main
# main()

# from src.generation.generation_all import main
# main()

# from src.evaluate_generator.evaluate_generator_all import main
# main()

# from src.analyze_split_point.analyze_split_point_all import main
# main()

# from src.analyze_retrieval.analyze_retrieval_all import main
# main()

# from src.analyze_embeddings.analyze_embeddings_all import main
# main()

# from pathlib import Path
# import os

# # przejdź do data/downloads/literaryQA
# os.chdir(
#     Path(__file__).resolve().parent / "data/downloads/literaryQA"
# )

# from pathlib import Path
# import sys

# ROOT = Path(__file__).resolve().parent / "data/downloads/literaryQA"

# sys.path.insert(0, str(ROOT))

# from data.downloads.literaryQA.scripts.download_and_clean_books import main
# from tap import Tap


# class ScriptArgs(Tap):
#     """Command-line arguments for the downloader script."""
#     output_dir: Path = Path("data/literaryqa")
#     write_as_jsonl: bool = False

#     def process_args(self) -> None:
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.logging_dir = self.output_dir / "logs"
#         self.logging_dir.mkdir(parents=True, exist_ok=True)

# main(ScriptArgs().parse_args())
