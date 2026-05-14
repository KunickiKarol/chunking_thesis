import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv

from src.tools.tokenizer_service import TokenizerService


class NovelQAPreprocessor:
    """
    Preprocessor dla datasetu novelQA.
    """

    SPLITS = ("train", "validation", "test", "example", "examples")

    def __init__(self, dataset_path: Path, params: dict):
        self.dataset_path = dataset_path
        self.params = params or {}

        self.cfg = self._load_env_config()

        self.downloads_dir: Path = self.cfg["downloads_dir"]
        self.random_state: int = self.cfg["random_state"]
        self.test_size: float = self.cfg["test_size"]
        self.val_size: float = self.cfg["val_size"]

        self.task_type: str = self.params.get("task_type")

        self.books_src = self.downloads_dir / "novelQA" / "Books"
        self.tasks_src = self.downloads_dir / "novelQA" / "Data"
        self.bookmeta_src = self.downloads_dir / "novelQA" / "bookmeta.json"

        if not self.bookmeta_src.exists():
            raise FileNotFoundError(f"Missing {self.bookmeta_src}")

    def run(self) -> Tuple[Path, Path, Path]:
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        books_dst, tasks_dst = self._prepare_split_dirs()

        bookmeta = self._load_bookmeta()
        split_map, n_test, n_val, n_train, example_map = self._compute_splits(list(bookmeta.keys()))

        self._annotate_bookmeta(bookmeta, split_map, example_map)
        self._save_bookmeta(bookmeta)

        docs_count, total_token_len = self._copy_books_by_split(
            self.books_src, books_dst, split_map, example_map, ".txt", do_tokenize=True
        )
        tasks_count, _ = self._copy_tasks_by_split(self.tasks_src, tasks_dst, split_map, example_map, ".json")

        print("✅ novelQA preprocessing finished")
        self._save_meta(total_token_len, docs_count, tasks_count, n_test, n_val, n_train)
        return books_dst, tasks_dst, self.dataset_path

    @staticmethod
    def _load_env_config() -> Dict:
        load_dotenv()
        return {
            "random_state": int(os.getenv("RANDOM_STATE")),
            "test_size": float(os.getenv("TEST_SIZE")),
            "val_size": float(os.getenv("VALIDATION_SIZE")),
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
            data = json.load(f)

        return {k: v for k, v in data.items() if v.get("copyright") == "PublicDomain"}

    def _compute_splits(self, book_ids: list[str]) -> Tuple[Dict[str, str], int, int, int, Dict[str, list[str]]]:
        random.seed(self.random_state)
        random.shuffle(book_ids)

        n_total = len(book_ids)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.val_size)

        split_map = {}

        for bookid in book_ids[:n_test]:
            split_map[bookid] = "test"

        for bookid in book_ids[n_test : n_test + n_val]:
            split_map[bookid] = "validation"

        for bookid in book_ids[n_test + n_val :]:
            split_map[bookid] = "train"

        train_ids = [bid for bid in book_ids if split_map.get(bid) == "train"]
        example_map = {
            "example": train_ids[:1],
            "examples": train_ids[:3],
        }

        return split_map, n_test, n_val, len(train_ids), example_map

    @staticmethod
    def _annotate_bookmeta(bookmeta: Dict, split_map: Dict[str, str], example_map: Dict[str, list[str]]):
        example_set = set(example_map.get("example", []))
        examples_set = set(example_map.get("examples", []))

        for bookid, meta in bookmeta.items():
            if isinstance(meta, dict):
                meta["split"] = split_map.get(bookid, "train")

                # flagi example / examples
                meta["example"] = bookid in example_set
                meta["examples"] = bookid in examples_set

    def _save_bookmeta(self, bookmeta: Dict):
        with open(self.dataset_path / "bookmeta.json", "w", encoding="utf-8") as f:
            json.dump(bookmeta, f, ensure_ascii=False, indent=2)

    def _save_meta(self, total_token_len, docs_count, tasks_count, n_test, n_val, n_train):
        meta = {
            "dataset_name": "novelQA",
            "task_type": "Multiple Choice",
            "docs_count": docs_count,
            "tasks_count": tasks_count,
            "avg_token_len": total_token_len // docs_count,
            "n_test": n_test,
            "n_val": n_val,
            "n_train": n_train,
        }

        with open(self.dataset_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _copy_books_by_split(
        self,
        src_dir: Path,
        dst_dir: Path,
        split_map: Dict[str, str],
        example_map: Dict[str, list[str]],
        extension: str,
        do_tokenize: bool = False,
    ):
        docs_count = 0
        total_token_len = 0

        if do_tokenize:
            tokenizer = TokenizerService()

        extra_splits = {}
        for split_name, ids in example_map.items():
            for bookid in ids:
                extra_splits.setdefault(bookid, []).append(split_name)

        for file_path in src_dir.rglob(f"*{extension}"):
            bookid = file_path.stem
            base_split = split_map.get(bookid, "train")
            docs_count += 1

            shutil.copy2(file_path, dst_dir / base_split / file_path.name)

            for extra_split in extra_splits.get(bookid, []):
                shutil.copy2(file_path, dst_dir / extra_split / file_path.name)

            if do_tokenize:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                total_token_len += tokenizer.tokenize(content)["token_count"]

        return docs_count, total_token_len

    def _copy_tasks_by_split(
        self,
        src_dir: Path,
        dst_dir: Path,
        split_map: Dict[str, str],
        example_map: Dict[str, list[str]],
        extension: str,
        do_tokenize: bool = False,
    ):
        docs_count = 0
        total_token_len = 0

        if do_tokenize:
            tokenizer = TokenizerService()

        extra_splits = {}
        for split_name, ids in example_map.items():
            for bookid in ids:
                extra_splits.setdefault(bookid, []).append(split_name)

        for file_path in src_dir.rglob(f"*{extension}"):
            bookid = file_path.stem
            base_split = split_map.get(bookid, "train")
            docs_count += 1

            # Wczytaj JSON
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Zamień Options dict -> list
            for question_data in data.values():
                if "Options" in question_data and isinstance(question_data["Options"], dict):
                    question_data["Options"] = list(question_data["Options"].values())

            # Zapisz do splitu bazowego
            output_path = dst_dir / base_split / file_path.name
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            # Zapisz do dodatkowych splitów
            for extra_split in extra_splits.get(bookid, []):
                output_path = dst_dir / extra_split / file_path.name
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            if do_tokenize:
                content = json.dumps(data, ensure_ascii=False)
                total_token_len += tokenizer.tokenize(content)["token_count"]

        return docs_count, total_token_len
