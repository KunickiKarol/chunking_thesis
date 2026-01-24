import os
import shutil
import random
import json
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv


class NovelQAPreprocessor:
    """
    Preprocessor dla datasetu NovelQA.
    """

    SPLITS = ("train", "validation", "test")

    def __init__(self, dataset_path: Path, params: dict):
        self.dataset_path = dataset_path
        self.params = params or {}

        self.cfg = self._load_env_config()

        self.downloads_dir: Path = self.cfg["downloads_dir"]
        self.random_state: int = self.cfg["random_state"]
        self.test_size: float = self.cfg["test_size"]
        self.val_size: float = self.cfg["val_size"]

        self.task_type: str = self.params.get("task_type")

        # źródła
        self.books_src = self.downloads_dir / "NovelQA" / "Books"
        self.tasks_src = self.downloads_dir / "NovelQA" / "Data"
        self.bookmeta_src = self.downloads_dir / "NovelQA" / "bookmeta.json"

        if not self.bookmeta_src.exists():
            raise FileNotFoundError(f"Missing {self.bookmeta_src}")

    # =========================
    # Public API
    # =========================

    def run(self) -> Tuple[Path, Path, Path]:
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        books_dst, tasks_dst = self._prepare_split_dirs()

        bookmeta = self._load_bookmeta()
        split_map = self._compute_splits(list(bookmeta.keys()))

        self._annotate_bookmeta(bookmeta, split_map)
        self._save_bookmeta(bookmeta)

        self._copy_files_by_split(self.books_src, books_dst, split_map, ".txt")
        self._copy_files_by_split(self.tasks_src, tasks_dst, split_map, ".json")

        print("✅ NovelQA preprocessing finished")

        return books_dst, tasks_dst, self.dataset_path


    @staticmethod
    def _load_env_config() -> Dict:
        load_dotenv()
        return {
            "random_state": int(os.getenv("RANDOM_STATE")),
            "test_size": float(os.getenv("NOVELQA_TEST_SIZE")),
            "val_size": float(os.getenv("NOVELQA_VALIDATION_SIZE")),
            "downloads_dir": Path(os.getenv("DOWNLOADS_DIR")),
        }

    def _prepare_split_dirs(self) -> Tuple[Path, Path]:
        books_dir = self.dataset_path / "Books"
        tasks_dir = self.dataset_path / "Tasks" / self.task_type

        for split in self.SPLITS:
            (books_dir / split).mkdir(parents=True, exist_ok=True)
            (tasks_dir / split).mkdir(parents=True, exist_ok=True)

        return books_dir, tasks_dir

    def _load_bookmeta(self) -> Dict:
        with open(self.bookmeta_src, encoding="utf-8") as f:
            return json.load(f)

    def _compute_splits(self, book_ids: list[str]) -> Dict[str, str]:
        random.seed(self.random_state)
        random.shuffle(book_ids)

        n_total = len(book_ids)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.val_size)

        split_map = {}

        for bookid in book_ids[:n_test]:
            split_map[bookid] = "test"

        for bookid in book_ids[n_test:n_test + n_val]:
            split_map[bookid] = "validation"

        for bookid in book_ids[n_test + n_val:]:
            split_map[bookid] = "train"

        return split_map

    @staticmethod
    def _annotate_bookmeta(bookmeta: Dict, split_map: Dict[str, str]):
        for bookid, meta in bookmeta.items():
            if isinstance(meta, dict):
                meta["split"] = split_map.get(bookid, "train")

    def _save_bookmeta(self, bookmeta: Dict):
        with open(self.dataset_path / "bookmeta.json", "w", encoding="utf-8") as f:
            json.dump(bookmeta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _copy_files_by_split(
        src_dir: Path,
        dst_dir: Path,
        split_map: Dict[str, str],
        extension: str,
    ):
        for file_path in src_dir.rglob(f"*{extension}"):
            bookid = file_path.stem
            split = split_map.get(bookid, "train")
            shutil.copy2(file_path, dst_dir / split / file_path.name)
