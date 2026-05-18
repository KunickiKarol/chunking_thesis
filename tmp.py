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

from src.analyze_split_point.analyze_split_point_all import main
main()

# from pathlib import Path

# from tap import Tap


# class ScriptArgs(Tap):
#     """Command-line arguments for the downloader script."""
#     output_dir: Path = Path("data/literaryqa")
#     write_as_jsonl: bool = False

#     def process_args(self) -> None:
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.logging_dir = self.output_dir / "logs"
#         self.logging_dir.mkdir(parents=True, exist_ok=True)
        
# from data.downloads.literaryQA.scripts.download_and_clean_books import main
# main(ScriptArgs().parse_args())