import json
import pandas as pd
from pathlib import Path
 
 
def load_analyze_embeddings(analyze_embeddings_dir: str | Path) -> pd.DataFrame:
    """
    Wczytuje wszystkie pliki results.json z drzewa katalogów o strukturze:
        ANALYZE_EMBEDDINGS_DIR/
            analyze_name/
                analyze_params_name/
                    .../
                        split_name/
                            results.json
 
    Każdy plik zawiera rekordy o kluczach w formacie:
        chunking_name-chunking_params_name-embed_name-embed_params_name
 
    Zwraca DataFrame z MultiIndex:
        (chunking_name, chunking_params_name, embed_name, embed_params_name,
         analyze_name, analyze_params_name)
 
    Kolumny: każdy klucz z 'metrics' jako osobna kolumna + 'label_counts'.
    """
    base = Path(analyze_embeddings_dir)
    records = []
 
    for results_file in base.rglob("results.json"):
        # Ścieżka względem base: analyze_name/analyze_params_name/.../results.json
        rel = results_file.relative_to(base)
        parts = rel.parts  # (analyze_name, analyze_params_name, ..., "results.json")
 
        if len(parts) < 3:
            # Za płytka ścieżka – pomijamy
            continue
 
        analyze_name = parts[0]
        analyze_params_name = parts[1]
        # Resztę katalogów (między analyze_params_name a results.json) ignorujemy
        # (split_name itp. nie jest wymagany w indeksie wg specyfikacji)
 
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
 
        for composite_key, value in data.items():
            key_parts = composite_key.split("-")
            if len(key_parts) != 4:
                raise ValueError(
                    f"Nieoczekiwany format klucza '{composite_key}' w pliku {results_file}. "
                    f"Oczekiwano 4 części rozdzielonych '-', otrzymano {len(key_parts)}."
                )
            chunking_name, chunking_params_name, embed_name, embed_params_name = key_parts
 
            row: dict = {
                "chunking_name": chunking_name,
                "chunking_params_name": chunking_params_name,
                "embed_name": embed_name,
                "embed_params_name": embed_params_name,
                "analyze_name": analyze_name,
                "analyze_params_name": analyze_params_name,
            }
 
            # Metryki jako osobne kolumny
            metrics = value.get("metrics", {})
            row.update(metrics)
 
            # label_counts jako jedna kolumna (słownik)
            row["label_counts"] = value.get("label_counts", {})
 
            records.append(row)
 
    if not records:
        # Zwróć pusty DataFrame z odpowiednimi kolumnami
        index_cols = [
            "chunking_name", "chunking_params_name",
            "embed_name", "embed_params_name",
            "analyze_name", "analyze_params_name",
        ]
        return pd.DataFrame(columns=index_cols + ["label_counts"])
 
    df = pd.DataFrame(records)
 
    index_cols = [
        "chunking_name",
        "chunking_params_name",
        "embed_name",
        "embed_params_name",
        "analyze_name",
        "analyze_params_name",
    ]
    df = df.set_index(index_cols).sort_index()
 
    # ------------------------------------------------------------------
    # Pivot: połącz wiersze o tym samym indeksie embedding (bez analyze_*)
    # Kolumny każdej grupy analyze_name/analyze_params_name dostają prefiks
    # f"{analyze_name}-{analyze_params_name}-{old_column_name}"
    # ------------------------------------------------------------------
    embed_index = ["chunking_name", "chunking_params_name", "embed_name", "embed_params_name"]
    analyze_index = ["analyze_name", "analyze_params_name"]
 
    parts = []
    for (analyze_name, analyze_params_name), group in df.groupby(level=analyze_index):
        # Zrzuć poziomy analyze_* z indeksu – zostaje sam indeks embedding
        group = group.droplevel(analyze_index)
        # Przemianuj wszystkie kolumny
        prefix = f"{analyze_name}-{analyze_params_name}"
        group = group.rename(columns=lambda col: f"{prefix}-{col}")
        parts.append(group)
 
    # join kolejno po indeksie embedding (outer żeby zachować wszystkie wiersze)
    df = parts[0].copy() if len(parts) == 1 else parts[0].join(parts[1:], how="outer")
    df = df.sort_index()
 
    # Usuń kolumny zawierające wyłącznie NaN
    df = df.dropna(axis=1, how="all")
 
    return df
