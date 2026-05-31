import json
from pathlib import Path

import numpy as np
import pandas as pd

TRACKED_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre"]


# ── Public entry point ────────────────────────────────────────────────────────

def load_analyze_split_point_dataframes(
    analyze_split_point_path: Path,
    configs: list[dict],
) -> pd.DataFrame:
    """
    configs = [
        {
            "dataset_name":               "...",
            "dataset_params_name":        "...",
            "chunking_name":              "...",
            "chunking_params_name":       "...",
            "analyze_retrieval_name":     "...",
            "analyze_retrieval_params_name": "...",
            "split_name":                 "...",
        },
        ...
    ]

    Zwraca DataFrame z MultiIndex:
        (dataset_name, dataset_params_name,
         chunking_name, chunking_params_name,
         analyze_retrieval_name, analyze_retrieval_params_name)

    Kolumny:
        Per tag  : {tag}_inside, {tag}_splited, {tag}_total,
                   {tag}_block_integrity_rate,
                   {tag}_avg_split_span, {tag}_max_split_span
        Globalne : block_integrity_rate
                   bpr_start, bpr_end
                   boundary_violation_count, boundary_violation_rate
                   orphaned_chunks, orphaned_chunk_rate
                   total_chunks, total_boundaries
        Zdania   : sentence_correct, sentence_incorrect,
                   sentence_total, sentence_correct_rate
    """
    # Grupuj configs po kluczu identyfikującym jeden wiersz w df
    # (wszystkie pola oprócz split_name)
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for cfg in configs:
        key = (
            cfg["dataset_name"],
            cfg["dataset_params_name"],
            cfg["chunking_name"],
            cfg["chunking_params_name"],
        )
        groups[key].append(cfg)

    rows = []
    for key, group_configs in groups.items():
        row = _build_row(analyze_split_point_path, key, group_configs)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index([
        "dataset_name",
        "dataset_params_name",
        "chunking_name",
        "chunking_params_name",
        "split_name"
    ]).sort_index()

    return df


# ── Per-group builder ─────────────────────────────────────────────────────────

def _build_row(
    base_path: Path,
    key: tuple,
    configs: list[dict],
) -> dict | None:
    """
    Agreguje wszystkie splity dla jednej kombinacji
    (dataset, params, chunker, chunker_params, analyze, analyze_params).
    """
    (
        dataset_name,
        dataset_params_name,
        chunking_name,
        chunking_params_name,
    ) = key

    # Akumulatory dla html_tag
    tag_acc = {
        tag: {"inside": 0, "splited": 0, "total": 0, "spans": []}
        for tag in TRACKED_TAGS
    }
    html_good_start = 0
    html_good_end = 0
    html_bvc = 0
    html_total_chunks = 0
    html_total_boundaries = 0
    html_orphaned_chunks = 0
    html_loaded = False

    # Akumulatory dla sentence
    sent_correct = 0
    sent_incorrect = 0
    sent_loaded = False

    for cfg in configs:
        split = cfg["split_name"]
        cfg_path = (
            base_path
            / dataset_name
            / dataset_params_name
            / chunking_name
            / chunking_params_name
        )

        # ── html_tag ──────────────────────────────────────────────────
        html_path = cfg_path / 'analyze_split_point' / "html_tag" / split / "analyze_split_point.json"
        if html_path.exists():
            html_data: dict = json.loads(html_path.read_text(encoding="utf-8"))
            for book_id, book in html_data.items():
                _accumulate_html(book, tag_acc)
                sp = book.get("split_point", {})
                html_good_start      += sp.get("good_start", {}).get("num", 0)
                html_good_end        += sp.get("good_end",   {}).get("num", 0)
                html_bvc             += sp.get("boundary_violation_count", 0)
                html_total_chunks    += book.get("total_chunks", 0)
                html_total_boundaries += book.get("total_boundaries", 0)
                cov = book.get("coverage", {})
                html_orphaned_chunks += cov.get("orphaned_chunks", 0)
            html_loaded = True
        else:
            print(f"Brak html_tag: {html_path}")

        # ── sentence ─────────────────────────────────────────────────
        sent_path = cfg_path / 'analyze_split_point' / "sentence" / split / "analyze_split_point.json"
        if sent_path.exists():
            sent_data: dict = json.loads(sent_path.read_text(encoding="utf-8"))
            for book_id, book in sent_data.items():
                sent_correct   += book.get("total_correct",   0)
                sent_incorrect += book.get("total_incorrect", 0)
            sent_loaded = True
        else:
            print(f"Brak sentence: {sent_path}")

    if not html_loaded and not sent_loaded:
        return None

    # ── Oblicz metryki per tag ────────────────────────────────────────
    row: dict = {}

    for tag in TRACKED_TAGS:
        acc = tag_acc[tag]
        inside  = acc["inside"]
        splited = acc["splited"]
        total   = acc["total"]
        spans   = acc["spans"]

        row[f"{tag}_inside"]  = inside
        row[f"{tag}_splited"] = splited
        row[f"{tag}_total"]   = total
        row[f"{tag}_block_integrity_rate"] = (
            inside / total if total > 0 else np.nan
        )
        row[f"{tag}_avg_split_span"] = (
            sum(spans) / len(spans) if spans else np.nan
        )
        row[f"{tag}_max_split_span"] = (
            max(spans) if spans else np.nan
        )

    # ── Globalne metryki html ────────────────────────────────────────
    total_inside_all = sum(tag_acc[t]["inside"] for t in TRACKED_TAGS)
    total_all        = sum(tag_acc[t]["total"]  for t in TRACKED_TAGS)

    row["block_integrity_rate"] = (
        total_inside_all / total_all if total_all > 0 else np.nan
    )
    row["bpr_start"] = (
        html_good_start / html_total_boundaries
        if html_total_boundaries > 0 else np.nan
    )
    row["bpr_end"] = (
        html_good_end / html_total_boundaries
        if html_total_boundaries > 0 else np.nan
    )
    row["boundary_violation_count"] = html_bvc
    row["boundary_violation_rate"] = (
        html_bvc / html_total_boundaries
        if html_total_boundaries > 0 else np.nan
    )
    row["orphaned_chunks"] = html_orphaned_chunks
    row["orphaned_chunk_rate"] = (
        html_orphaned_chunks / html_total_chunks
        if html_total_chunks > 0 else np.nan
    )
    row["total_chunks"]     = html_total_chunks
    row["total_boundaries"] = html_total_boundaries

    # ── Metryki sentence ─────────────────────────────────────────────
    sent_total = sent_correct + sent_incorrect
    row["sentence_correct"]      = sent_correct
    row["sentence_incorrect"]    = sent_incorrect
    row["sentence_total"]        = sent_total
    row["sentence_correct_rate"] = (
        sent_correct / sent_total if sent_total > 0 else np.nan
    )

    # ── Indeks ───────────────────────────────────────────────────────
    row["dataset_name"]                    = dataset_name
    row["dataset_params_name"]             = dataset_params_name
    row["chunking_name"]                   = chunking_name
    row["chunking_params_name"]            = chunking_params_name
    row['split_name']                      = split
    return row


# ── Helper ────────────────────────────────────────────────────────────────────

def _accumulate_html(book: dict, tag_acc: dict) -> None:
    """Dołącz dane jednej książki do akumulatorów tagów."""
    for tag in TRACKED_TAGS:
        if tag not in book:
            continue
        t = book[tag]
        tag_acc[tag]["inside"]  += t.get("inside",  {}).get("num", 0)
        tag_acc[tag]["splited"] += t.get("splited", {}).get("num", 0)
        tag_acc[tag]["total"]   += t.get("total", 0)
        # split_chunk_spans to lista list (jedna na split) – spłaszczamy
        tag_acc[tag]["spans"].extend(t.get("split_chunk_spans", []))