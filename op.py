
#!/usr/bin/env python3
"""
Skrypt przetwarzający foldery rerank_result.

Dla każdego folderu pasującego do wzoru:
  .../faiss/GGG/reranker/HHH/III

Tworzy odpowiadający folder:
  .../faiss/GGG/reranker/HHH_op/III

z:
  - meta.json         — skopiowany bez zmian
  - rerank_results.json — ID posortowane rosnąco dla każdego klucza
  - scores_results.json — score'y przepisane zgodnie z nowym przypisaniem score→ID
"""

import json
import shutil
import sys
from pathlib import Path


BASE_DIR = Path("/home/mahuk/chunking-thesis/data/rerank_result")


def find_source_dirs(base: Path) -> list[Path]:
    """
    Znajduje wszystkie katalogi zawierające trzy wymagane pliki JSON,
    pomijając katalogi których komponent HHH już zawiera '_op'.
    """
    found = []
    for meta in base.rglob("meta.json"):
        d = meta.parent
        # Pomijamy katalogi już przetworzone (_op)
        # Sprawdzamy czy któryś element ścieżki względem base zawiera "_op"
        rel = d.relative_to(base)
        parts = rel.parts
        # Wzorzec: AAA/BBB/CCC/DDD/EEE/FFF/faiss/GGG/reranker/HHH/III
        # Szukamy segmentu 'reranker' i sprawdzamy HHH
        if "reranker" not in parts:
            continue
        reranker_idx = parts.index("reranker")
        if reranker_idx + 1 >= len(parts):
            continue
        hhh = parts[reranker_idx + 1]
        if hhh.endswith("_op"):
            continue  # to już jest katalog wynikowy, pomijamy

        # Sprawdzamy czy są wymagane pliki
        if not (d / "rerank_results.json").exists():
            continue
        if not (d / "scores_results.json").exists():
            continue

        found.append(d)
    return found


def build_target_dir(source: Path, base: Path) -> Path:
    """
    Dla ścieżki źródłowej buduje ścieżkę docelową,
    dodając '_op' do segmentu HHH (tuż za 'reranker').
    """
    rel = source.relative_to(base)
    parts = list(rel.parts)
    reranker_idx = parts.index("reranker")
    hhh_idx = reranker_idx + 1
    parts[hhh_idx] = parts[hhh_idx] + "_op"
    return base / Path(*parts)


def process_dir(source: Path, target: Path) -> None:
    """Przetwarza jeden katalog źródłowy → docelowy."""

    # Wczytaj dane źródłowe
    with open(source / "rerank_results.json", encoding="utf-8") as f:
        rerank: dict[str, list[int]] = json.load(f)

    with open(source / "scores_results.json", encoding="utf-8") as f:
        scores: dict[str, list[float]] = json.load(f)

    # Buduj nowe słowniki
    new_rerank: dict[str, list[int]] = {}
    new_scores: dict[str, list[float]] = {}

    for key in rerank:
        ids = rerank[key]
        sc = scores[key]

        if len(ids) != len(sc):
            print(f"  UWAGA: dla klucza '{key}' liczba ID ({len(ids)}) "
                  f"!= liczba scores ({len(sc)}) — pomijam klucz.")
            new_rerank[key] = ids
            new_scores[key] = sc
            continue

        # Tworzymy pary (id, score) i sortujemy po id rosnąco
        pairs = sorted(zip(ids, sc), key=lambda x: x[0])

        new_rerank[key] = [p[0] for p in pairs]
        new_scores[key] = [p[1] for p in pairs]

    # Utwórz katalog docelowy
    target.mkdir(parents=True, exist_ok=True)

    # Zapisz meta.json (kopia bez zmian)
    shutil.copy2(source / "meta.json", target / "meta.json")

    # Zapisz zmodyfikowany rerank_results.json
    with open(target / "rerank_results.json", "w", encoding="utf-8") as f:
        json.dump(new_rerank, f, indent=2, ensure_ascii=False)

    # Zapisz zmodyfikowany scores_results.json
    with open(target / "scores_results.json", "w", encoding="utf-8") as f:
        json.dump(new_scores, f, indent=2, ensure_ascii=False)


def main() -> None:
    if not BASE_DIR.exists():
        print(f"BŁĄD: Katalog bazowy nie istnieje: {BASE_DIR}")
        sys.exit(1)

    source_dirs = find_source_dirs(BASE_DIR)

    if not source_dirs:
        print("Nie znaleziono żadnych katalogów do przetworzenia.")
        return

    print(f"Znaleziono {len(source_dirs)} katalog(ów) do przetworzenia.\n")

    skipped = 0
    processed = 0

    for source in sorted(source_dirs):
        target = build_target_dir(source, BASE_DIR)
        rel_source = source.relative_to(BASE_DIR)

        if target.exists():
            print(f"[POMINIĘTO]  {rel_source}")
            print(f"             → {target.relative_to(BASE_DIR)} już istnieje\n")
            skipped += 1
            continue

        print(f"[PRZETWARZA] {rel_source}")
        print(f"             → {target.relative_to(BASE_DIR)}")
        try:
            process_dir(source, target)
            print(f"             ✓ Gotowe\n")
            processed += 1
        except Exception as e:
            print(f"             ✗ BŁĄD: {e}\n")

    print("─" * 60)
    print(f"Przetworzone: {processed}  |  Pominięte (już istnieją): {skipped}")


if __name__ == "__main__":
    main()