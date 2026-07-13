from src.tools.logging_config import setup_logging

setup_logging()

# from src.download_datasets.download_datasets_all import main
# main()

# from src.preprocess_datasets.preprocess_datasets_all import main
# main()

# from src.chunking.chunking_text_all import main
# main()

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

# from src.analyze_split_point.analyze_split_point_all import 
# main
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
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import yaml

from src.tools.presets import get_list_of_presets
from src.tools.read.rerank.load_rerank_raw import load_retrieval_dataframes


load_dotenv()

RERANK_DIR = Path(os.getenv("RERANK_DIR"))

with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

datasets_cfg = params.get("preprocess_datasets").get("datasets")
if not datasets_cfg:
    raise ValueError("Nie znaleziono datasets_download w params.yaml")

chunking_cfg = params.get("chunking").get("methods")
if not chunking_cfg:
    raise ValueError("Nie znaleziono chunking_methods w params.yaml")

embed_cfg = params.get("vector_embed").get("methods")
if not embed_cfg:
    raise ValueError("Nie znaleziono vector_embed_methods w params.yaml")

search_cfg = params.get("search").get("methods")
if not embed_cfg:
    raise ValueError("Nie znaleziono search methods w params.yaml")

rerank_cfg = params.get("rerank").get("methods")
if not embed_cfg:
    raise ValueError("Nie znaleziono rerank methods w params.yaml")

analyze_cfg = params.get("analyze_retrieval").get("methods")
if not analyze_cfg:
    raise ValueError("Nie znaleziono analyze_retrieval_methods w params.yaml")

splits = params.get("chunking").get("splits")
if not splits:
    raise ValueError("Nie znaleziono splits w params.yaml")

dataset_names, dataset_params_names = get_list_of_presets(datasets_cfg)
chunking_names, chunking_params_names = get_list_of_presets(chunking_cfg)
embed_names, embed_params_names = get_list_of_presets(embed_cfg)
search_names, search_params_names = get_list_of_presets(search_cfg)
rerank_names, rerank_params_names = get_list_of_presets(rerank_cfg)

configs = []
for dataset, dataset_params in zip(dataset_names, dataset_params_names):
    for chunking, chunking_params in zip(chunking_names, chunking_params_names):
        for embed, embed_params in zip(embed_names, embed_params_names):
            if 'local' in embed_params:
                continue
            for search, search_params in zip(search_names, search_params_names):
                for rerank, rerank_params in zip(rerank_names, rerank_params_names):
                    for split in splits:
                        configs.append({
                            "dataset_name": dataset,
                            "dataset_params_name": dataset_params,
                            "chunking_name": chunking,
                            "chunking_params_name": chunking_params,
                            "embed_name": embed,
                            "embed_params_name": embed_params,
                            "search_name": search,
                            "search_params_name": search_params,
                            "rerank_name": rerank,
                            "rerank_params_name": rerank_params,
                            "split_name": split
                        })
retrieval_df = load_retrieval_dataframes(RERANK_DIR, configs)
retrieval_df