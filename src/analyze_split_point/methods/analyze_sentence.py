import json
from pathlib import Path
import re
from collections import defaultdict
import time
from src.analyze_split_point.methods.register import register_analyze_split_point

SENTENCE_END_REGEX = re.compile(r"[\.!\?…]")
DIALOG_BOUNDARY_REGEX = re.compile(r'[:;,\.\?!]\s*[-—]\s*$')

import re

SENTENCE_END_REGEX = re.compile(r"[\.!\?…]")

DIALOG_BOUNDARY_REGEX = re.compile(
    r'[:;,\.\?!]\s*[-—]["\'»”’\)\]\}]*\s*$'
)


def is_good_boundary(prev_text: str, between_text: str, next_text: str) -> bool:
    """
    Heurystyka oceny poprawności splitu.
    """

    if not prev_text or not next_text:
        return False

    between_text = between_text or ""

    # ---------------------------------
    # 1. Koniec zdania
    # ---------------------------------

    prev_ends_sentence = bool(
        re.search(
            r'[\.!\?…]["\'»”’\)\]\}]*\s*$',
            prev_text
        )
    )

    # ---------------------------------
    # 2. Boundary dialogowe typu :—
    # ---------------------------------

    prev_has_dialog_boundary = bool(
        DIALOG_BOUNDARY_REGEX.search(prev_text)
    )

    # ---------------------------------
    # 3. Czy między chunkami są tylko neutralne znaki
    # ---------------------------------

    clean_between = re.sub(
        r'[\s"\'„”«»\(\)\[\]\{\}-—]+',
        "",
        between_text
    )

    between_is_clean = clean_between == ""



    # ---------------------------------
    # 5. Czy next wygląda jak początek zdania/dialogu
    # ---------------------------------

    next_starts_sentence_like = bool(
        re.match(
            r'^[\s"\'„”«»\(\[]*[A-ZĄĆĘŁŃÓŚŹŻ]',
            next_text
        )
    )

    # ---------------------------------
    # HARD FAIL
    # ---------------------------------

    if not between_is_clean:
        return False


    # ---------------------------------
    # POPRAWNE SPLITY
    # ---------------------------------

    # klasyczny koniec zdania
    if prev_ends_sentence and next_starts_sentence_like:
        return True

    # dialogi typu :—
    if prev_has_dialog_boundary and next_starts_sentence_like:
        return True

    # paragraph split
    if "\n\n" in between_text:
        return True

    # 1345 LiteraryQA
    if (
        prev_text.endswith("!")
        and between_text == ' '
        and next_text
        and next_text[0].islower()
    ):
        return True
    
    if re.search(r'\n[IVXLCDM]+$', prev_text) and between_text == '\n':
        return True
    
    if prev_text.endswith('—"'):
        return True
    
    if prev_text[-1] == ';' and between_text == ' ' and next_text[0] in ("'", '"'):
        return True
    if next_text.startswith("—"):
        return True
    # if next_text.startswith((". ", "! ", "? ", ":\n", ".\n", "?\n", "!\n")) and between_text == "" and prev_text[-1].islower():
    #     return True
    if prev_text.endswith((":")) and between_text in (" ", "\n") and next_text[0].isupper():
        return True
    return False


@register_analyze_split_point("sentence")
def analyze_sentence(chunks_files, books_files, tags_files, analyze_preset_params):
    start_time = time.perf_counter()
    books_map = {f.stem: f for f in books_files}


    chunks_analyze = {}

    for chunk_file in chunks_files:
        total_correct = 0
        incorrect_after_chunk_id = []
        book_id = chunk_file.stem
        if book_id not in books_map:
            continue

        book_text = books_map[book_id].read_text(encoding="utf-8")

        chunks = []
        with chunk_file.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        chunks.sort(key=lambda x: x["chunk_id"])

        for i in range(len(chunks) - 1):
            c1 = chunks[i]
            c2 = chunks[i + 1]

            start1, end1 = c1["start_index"], c1["end_index"]
            start2, end2 = c2["start_index"], c2["end_index"]

            # tekst chunków
            prev_text = book_text[start1:end1 + 1]
            next_text = book_text[start2:end2 + 1]

            # fragment między chunkami (może być luka np. +5 indeksów)
            between_text = ""
            if end1 + 1 < start2:
                between_text = book_text[end1 + 1:start2]

            is_correct = is_good_boundary(prev_text, between_text, next_text)

            if is_correct:
                total_correct += 1
            else:
                incorrect_after_chunk_id.append(c1["chunk_id"])
        chunks_analyze[book_id] = {
            "total_correct": total_correct,
            "total_incorrect": len(incorrect_after_chunk_id),
            "incorrect_after_chunk_id": incorrect_after_chunk_id,
        }
    total_time = time.perf_counter() - start_time
    return chunks_analyze, total_time
