import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv

from src.tools.models_cache import get_tokenizer_service

logger = logging.getLogger(__name__)


class infiniteBenchChoicePreprocessor:
    """
    Preprocessor dla datasetu infiniteBench Choice.
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
        self.books_src = self.downloads_dir / "infiniteBench" / "longbook_choice_eng.jsonl"

    # =========================
    # Public API
    # =========================

    def run(self) -> Tuple[Path, Path, Path]:
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        books_dst, tasks_dst = self._prepare_split_dirs()

        bookmeta, total_token_len = self._load_bookmeta()
        split_map, old_id_split_map, n_test, n_val, n_train, example_map = self._compute_splits(bookmeta)

        self._annotate_bookmeta(bookmeta, split_map, example_map)
        self._save_bookmeta(bookmeta)

        bookid_to_contentid, docs_count = self._create_books_files(self.books_src, books_dst, split_map, example_map)

        tasks_count = self._create_tasks_files(
            self.books_src, tasks_dst, split_map, old_id_split_map, bookid_to_contentid, example_map
        )

        self._save_meta(total_token_len, docs_count, tasks_count, n_test, n_val, n_train)

        logger.info("✅ infiniteBenchQA preprocessing finished")

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
        metadata = {}
        tokenizer = get_tokenizer_service()

        docs_count = 0
        total_token_len = 0

        with open(self.books_src, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                old_id = obj.get("id")
                if old_id is None:
                    continue

                content = obj.get("context", "")
                content_token_len = tokenizer.tokenize(content)["token_count"]

                docs_count += 1
                total_token_len += content_token_len

                metadata[old_id] = {
                    "old_id": old_id,
                    "title": content[:20],
                    "source_dataset": "infinityBench",
                    "tokenlen": content_token_len,
                    "charlen": len(content),
                }

        deduplicated = {}
        seen = {}

        for value in metadata.values():
            # ignorujemy old_id przy deduplikacji
            compare_value = {k: v for k, v in value.items() if k != "old_id"}

            value_key = json.dumps(compare_value, sort_keys=True, ensure_ascii=False)

            if value_key not in seen:
                new_key = str(len(seen))
                seen[value_key] = new_key

                deduplicated[new_key] = value
            else:
                # jeśli duplikat -> dopisz stare id
                existing_key = seen[value_key]

                existing_old_id = deduplicated[existing_key]["old_id"]

                if isinstance(existing_old_id, list):
                    existing_old_id.append(value["old_id"])
                else:
                    deduplicated[existing_key]["old_id"] = [existing_old_id, value["old_id"]]

        return deduplicated, total_token_len

    def _compute_splits(self, bookmeta: Dict[str, dict]) -> Tuple[
        Dict[str, str],  # split map po nowych id
        Dict[str, str],  # split map po old_id
        int,
        int,
        int,
        Dict[str, list[str]],
    ]:
        # splitujemy tylko po nowych deduplikowanych id
        book_ids = list(bookmeta.keys())

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

        # mapowanie old_id -> split
        old_id_split_map = {}

        for bookid, split in split_map.items():
            old_ids = bookmeta[bookid]["old_id"]

            if not isinstance(old_ids, list):
                old_ids = [old_ids]

            for old_id in old_ids:
                old_id_split_map[old_id] = split

        train_ids = [bid for bid in book_ids if split_map.get(bid) == "train"]

        example_map = {
            "example": train_ids[:1],
            "examples": train_ids[:3],
        }

        return (
            split_map,
            old_id_split_map,
            n_test,
            n_val,
            len(train_ids),
            example_map,
        )

    @staticmethod
    def _annotate_bookmeta(
        bookmeta: Dict,
        split_map: Dict[str, str],
        example_map: Dict[str, list[str]],
    ):
        example_set = set(example_map.get("example", []))
        examples_set = set(example_map.get("examples", []))

        for bookid, meta in bookmeta.items():
            if isinstance(meta, dict):
                meta["split"] = split_map.get(bookid, "train")

                # 🔥 nowe flagi
                meta["example"] = bookid in example_set
                meta["examples"] = bookid in examples_set

    def _save_bookmeta(self, bookmeta: Dict):
        with open(self.dataset_path / "bookmeta.json", "w", encoding="utf-8") as f:
            json.dump(bookmeta, f, ensure_ascii=False, indent=2)

    def _save_meta(self, total_token_len, docs_count, tasks_count, n_test, n_val, n_train):
        meta = {
            "dataset_name": "infiniteBenchChoice",
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

    @staticmethod
    def _create_books_files(
        src_dir: Path,
        dst_dir: Path,
        split_map: Dict[str, str],
        example_map: Dict[str, list[str]],
    ):
        content_to_id = {}
        bookid_to_contentid = {}

        next_id = 0
        docs_count = 0

        extra_splits = {}
        for split_name, ids in example_map.items():
            for bookid in ids:
                extra_splits.setdefault(bookid, []).append(split_name)

        with open(src_dir, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                bookid = obj.get("id")
                content = obj.get("context", "")

                if bookid is None or not content:
                    continue

                # deduplikacja po content
                if content in content_to_id:
                    content_id = content_to_id[content]
                else:
                    content_id = str(next_id)
                    content_to_id[content] = content_id
                    next_id += 1

                bookid_to_contentid[bookid] = content_id

                split = split_map.get(content_id)
                if split is None:
                    continue

                # normal split
                split_dir = dst_dir / split
                split_dir.mkdir(parents=True, exist_ok=True)

                file_path = split_dir / f"{content_id}.txt"

                if not file_path.exists():
                    with open(file_path, "w", encoding="utf-8") as out:
                        out.write(content)
                    docs_count += 1

                # example/examples
                for extra_split in extra_splits.get(content_id, []):
                    extra_split_path = dst_dir / extra_split
                    extra_split_path.mkdir(parents=True, exist_ok=True)

                    extra_file = extra_split_path / f"{content_id}.txt"

                    if not extra_file.exists():
                        with open(extra_file, "w", encoding="utf-8") as out:
                            out.write(content)

        return bookid_to_contentid, docs_count

    @staticmethod
    def _create_tasks_files(
        src_dir: Path,
        dst_dir: Path,
        split_map: Dict[str, str],
        old_id_split_map: Dict[str, str],
        bookid_to_contentid: Dict[str, str],
        example_map: Dict[str, list[str]],
    ):
        tasks_per_content: Dict[str, Dict[str, Any]] = {}

        # content_id -> ["example", "examples"]
        extra_splits = {}

        for split_name, ids in example_map.items():
            for dedup_bookid in ids:
                extra_splits.setdefault(dedup_bookid, []).append(split_name)

        with open(src_dir, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                old_id = obj.get("id")

                if old_id is None:
                    continue

                content_id = bookid_to_contentid.get(old_id)

                if content_id is None:
                    continue

                split = old_id_split_map.get(old_id)

                if split is None:
                    continue

                if content_id not in tasks_per_content:
                    tasks_per_content[content_id] = {"split": split, "old_ids": set(), "data": {}}

                tasks_per_content[content_id]["old_ids"].add(old_id)

                q_key = f"Q{old_id}"

                tasks_per_content[content_id]["data"][q_key] = {
                    "Question": obj.get("input", ""),
                    "Answers": obj.get("answer", ""),
                    "Options": obj.get("options", ""),
                }

        tasks_count = 0

        for content_id, info in tasks_per_content.items():
            split = info["split"]
            data = info["data"]

            # normal split
            split_dir = dst_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            with open(split_dir / f"{content_id}.json", "w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False, indent=2)

            tasks_count += len(data.keys())

            # extra split
            matching_splits = extra_splits.get(content_id, [])

            for split_name in matching_splits:
                extra_dir = dst_dir / split_name
                extra_dir.mkdir(parents=True, exist_ok=True)

                with open(extra_dir / f"{content_id}.json", "w", encoding="utf-8") as out:
                    json.dump(data, out, ensure_ascii=False, indent=2)

        return tasks_count
