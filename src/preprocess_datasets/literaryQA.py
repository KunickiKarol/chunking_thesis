import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from src.tools.tokenizer_service import TokenizerService


class LiteraryQAPreprocessor:
    """
    Preprocessor dla datasetu literaryQA.
    """

    def __init__(self, dataset_path: Path, params: dict):
        self.dataset_path = dataset_path
        self.params = params

        self.task_type = self.params.get("task_type")

        self.downloads_dir = self._load_env_config()

        self.data_src = self.downloads_dir / "literaryQA" / "data"
        self.books_src = self.data_src / "literaryqa"
        self.annotations_src = self.data_src / "annotations"

    def run(self) -> Tuple[Path, Path, Path]:
        books_dst, tasks_dst, bookmeta_dst, meta_path = self._prepare_output_dirs()

        (
            docs_count,
            total_token_len,
            n_train,
            n_test,
            n_val,
            book_token_lengths,
            train_books_order,
            book_path_map,
        ) = self._copy_books(self.books_src, books_dst)

        print(f"📚 Skopiowano książki do {books_dst}")

        bookmeta, tasks_count, tasks_by_book, task_file_map = self._process_annotations(
            self.annotations_src,
            tasks_dst,
            book_token_lengths,
        )

        self._create_additional_splits(
            books_dst=books_dst,
            tasks_dst=tasks_dst,
            train_books_order=train_books_order,
            book_path_map=book_path_map,
            task_file_map=task_file_map,
            bookmeta=bookmeta,
        )

        with bookmeta_dst.open("w", encoding="utf-8") as f:
            json.dump(bookmeta, f, ensure_ascii=False, indent=4)

        self._save_meta(
            meta_path,
            total_token_len,
            docs_count,
            tasks_count,
            n_train,
            n_test,
            n_val,
        )

        print("✅ Preprocessing literaryQA zakończony")

        return books_dst, tasks_dst, bookmeta_dst

    @staticmethod
    def _load_env_config() -> Path:
        load_dotenv()
        return Path(os.getenv("DOWNLOADS_DIR"))

    def _prepare_output_dirs(self) -> Tuple[Path, Path, Path, Path]:
        base_dir = self.dataset_path

        books_dir = base_dir / "Books"
        tasks_dir = base_dir / "Tasks" / self.task_type
        bookmeta_path = base_dir / "bookmeta.json"
        meta_path = base_dir / "meta.json"

        books_dir.mkdir(parents=True, exist_ok=True)
        tasks_dir.mkdir(parents=True, exist_ok=True)

        return books_dir, tasks_dir, bookmeta_path, meta_path

    def _save_meta(
        self,
        meta_path,
        total_token_len,
        docs_count,
        tasks_count,
        n_train,
        n_test,
        n_val,
    ):
        meta = {
            "dataset_name": "literaryQA",
            "task_type": "Short Answer",
            "docs_count": docs_count,
            "tasks_count": tasks_count,
            "avg_token_len": total_token_len // docs_count,
            "n_test": n_test,
            "n_val": n_val,
            "n_train": n_train,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _copy_books(src_dir: Path, dst_dir: Path):
        total_token_len = 0
        docs_count = 0

        n_train = 0
        n_test = 0
        n_val = 0

        book_token_lengths = {}
        train_books_order: List[str] = []
        book_path_map: Dict[str, Path] = {}

        tokenizer = TokenizerService()

        for txt_file in src_dir.rglob("*.cleaned.txt"):
            relative_path = txt_file.relative_to(src_dir)
            new_name = txt_file.name.replace(".cleaned", "")

            split = relative_path.parts[0]

            book_id = txt_file.stem.replace(".cleaned", "")

            if split == "train":
                n_train += 1
                train_books_order.append(book_id)

            elif split == "test":
                n_test += 1
            elif split == "validation":
                n_val += 1

            dest_file = dst_dir / relative_path.parent / new_name
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(txt_file, dest_file)

            book_path_map[book_id] = dest_file

            docs_count += 1

            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            token_count = tokenizer.tokenize(content)["token_count"]
            total_token_len += token_count

            book_token_lengths[book_id] = token_count

        return (
            docs_count,
            total_token_len,
            n_train,
            n_test,
            n_val,
            book_token_lengths,
            train_books_order,
            book_path_map,
        )

    @staticmethod
    def _process_annotations(
        annotations_dir: Path,
        tasks_dir: Path,
        book_token_lengths: Dict[str, int],
    ) -> Tuple[Dict[str, dict], int, Dict[str, dict], Dict[str, Path]]:

        bookmeta = {}
        tasks_by_book: Dict[str, dict] = {}
        task_file_map: Dict[str, Path] = {}

        q_counter = 1
        tasks_count = 0

        tokenizer = TokenizerService()

        for json_file in annotations_dir.rglob("*.jsonl"):
            with json_file.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)

                    book_id = str(record.get("gutenberg_id"))
                    summary = record.get("summary") or ""

                    bookmeta[book_id] = {
                        "title": record.get("title"),
                        "author": record.get("metadata", {}).get("author"),
                        "split": record.get("split"),
                        "publication_date": record.get("metadata", {}).get("publication_date"),
                        "genre_tags": record.get("metadata", {}).get("genre_tags"),
                        "text_url": record.get("metadata", {}).get("text_url"),
                        "summary_url": record.get("metadata", {}).get("summary_url"),
                        "summary": summary,
                        "summary_token_len": tokenizer.tokenize(summary)["token_count"],
                        "tokenlen": book_token_lengths.get(book_id),
                        # 🔥 DODAJ TO:
                        "example": False,
                        "examples": False,
                    }

                    tasks = {}

                    for qa in record.get("qas", []):
                        q_id = f"Q{q_counter:04d}"
                        q_counter += 1

                        tasks[q_id] = {
                            "Question": qa.get("question"),
                            "Answers": qa.get("answers", []),
                        }

                    tasks_count += 1
                    tasks_by_book[book_id] = tasks

                    out_file = tasks_dir / record.get("split") / f"{book_id}.json"
                    out_file.parent.mkdir(parents=True, exist_ok=True)

                    with out_file.open("w", encoding="utf-8") as out:
                        json.dump(tasks, out, ensure_ascii=False, indent=4)

                    task_file_map[book_id] = out_file

        return bookmeta, tasks_count, tasks_by_book, task_file_map

    # ---------------- SPLITY ----------------

    def _create_additional_splits(
        self,
        books_dst: Path,
        tasks_dst: Path,
        train_books_order: List[str],
        book_path_map: Dict[str, Path],
        task_file_map: Dict[str, Path],
        bookmeta: Dict[str, dict],
    ):
        example_books_dir = books_dst / "example"
        examples_books_dir = books_dst / "examples"

        example_tasks_dir = tasks_dst / "example"
        examples_tasks_dir = tasks_dst / "examples"

        example_books_dir.mkdir(parents=True, exist_ok=True)
        examples_books_dir.mkdir(parents=True, exist_ok=True)
        example_tasks_dir.mkdir(parents=True, exist_ok=True)
        examples_tasks_dir.mkdir(parents=True, exist_ok=True)

        example_ids = set(train_books_order[:1])
        examples_ids = set(train_books_order[:3])

        def copy(book_id: str, books_target: Path, tasks_target: Path):
            # BOOKS
            src_book = book_path_map.get(book_id)
            if src_book and src_book.exists():
                dest_book = books_target / src_book.name
                dest_book.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_book, dest_book)

            # TASKS
            src_task = task_file_map.get(book_id)
            if src_task and src_task.exists():
                dest_task = tasks_target / src_task.name
                dest_task.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_task, dest_task)

            # METADATA FLAGS
            meta = bookmeta.get(book_id)
            if meta is not None:
                meta.setdefault("metadata", {})

                if book_id in example_ids:
                    meta["example"] = True

                if book_id in examples_ids:
                    meta["examples"] = True

        # example = 1 książka train
        for book_id in example_ids:
            copy(book_id, example_books_dir, example_tasks_dir)

        # examples = 3 książki train
        for book_id in examples_ids:
            copy(book_id, examples_books_dir, examples_tasks_dir)
