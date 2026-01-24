import os
import shutil
import json
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv


class LiteraryQAPreprocessor:
    """
    Preprocessor dla datasetu LiteraryQA.
    """

    def __init__(self, dataset_path: Path, params: dict):
        self.dataset_path = dataset_path
        self.params = params

        self.task_type = self.params.get("task_type")

        self.downloads_dir = self._load_env_config()

        # źródła
        self.data_src = self.downloads_dir / "LiteraryQA" / "data"
        self.books_src = self.data_src / "literaryqa"
        self.annotations_src = self.data_src / "annotations"


    def run(self) -> Tuple[Path, Path, Path]:
        books_dst, tasks_dst, bookmeta_dst = self._prepare_output_dirs()

        self._copy_books(self.books_src, books_dst)
        print(f"📚 Skopiowano książki do {books_dst}")

        bookmeta = self._process_annotations(self.annotations_src, tasks_dst)

        bookmeta_dst.parent.mkdir(parents=True, exist_ok=True)
        with bookmeta_dst.open("w", encoding="utf-8") as f:
            json.dump(bookmeta, f, ensure_ascii=False, indent=4)

        print(f"🧾 Zapisano metadata do {bookmeta_dst}")
        print("✅ Preprocessing LiteraryQA zakończony")

        return books_dst, tasks_dst, bookmeta_dst


    @staticmethod
    def _load_env_config() -> Path:
        load_dotenv()
        return Path(os.getenv("DOWNLOADS_DIR"))

    def _prepare_output_dirs(self) -> Tuple[Path, Path, Path]:
        base_dir = self.dataset_path

        books_dir = base_dir / "Books"
        tasks_dir = base_dir / "Tasks" / self.task_type
        bookmeta_path = base_dir / "bookmeta.json"

        books_dir.mkdir(parents=True, exist_ok=True)
        tasks_dir.mkdir(parents=True, exist_ok=True)

        return books_dir, tasks_dir, bookmeta_path

    @staticmethod
    def _copy_books(src_dir: Path, dst_dir: Path):
        for txt_file in src_dir.rglob("*.cleaned.txt"):
            relative_path = txt_file.relative_to(src_dir)
            new_name = txt_file.name.replace(".cleaned", "")

            dest_file = dst_dir / relative_path.parent / new_name
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(txt_file, dest_file)

    @staticmethod
    def _process_annotations(
        annotations_dir: Path,
        tasks_dir: Path,
    ) -> Dict[str, dict]:
        bookmeta = {}
        q_counter = 1

        for json_file in annotations_dir.rglob("*.jsonl"):
            with json_file.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)

                    gutenberg_id = record.get("gutenberg_id")

                    # ---- metadata ----
                    bookmeta[str(gutenberg_id)] = {
                        "title": record.get("title"),
                        "author": record.get("metadata", {}).get("author"),
                        "split": record.get("split"),
                        "publication_date": record.get("metadata", {}).get("publication_date"),
                        "genre_tags": record.get("metadata", {}).get("genre_tags"),
                        "text_url": record.get("metadata", {}).get("text_url"),
                        "summary_url": record.get("metadata", {}).get("summary_url"),
                        "summary": record.get("summary"),
                    }

                    # ---- QAS ----
                    tasks = {}
                    for qa in record.get("qas", []):
                        q_id = f"Q{q_counter:04d}"
                        q_counter += 1

                        tasks[q_id] = {
                            "Question": qa.get("question"),
                            "Answers": qa.get("answers", []),
                        }

                    # ---- save ----
                    out_dir = tasks_dir / json_file.stem
                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_file = out_dir / f"{gutenberg_id}.json"
                    with out_file.open("w", encoding="utf-8") as out:
                        json.dump(tasks, out, ensure_ascii=False, indent=4)

        return bookmeta
