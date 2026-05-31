import json
import time
from collections import defaultdict
 
from src.analyze_split_point.methods.register import register_analyze_split_point
 
 
@register_analyze_split_point("html_tag")
def analyze_html_tags(chunks_files, books_files, tags_files, analyze_preset_params):
    start_time = time.perf_counter()
    tags_map = {f.stem: f for f in tags_files}
    chunks_analyze = {}
 
    TRACKED_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre"]
 
    for chunk_file in chunks_files:
        book_id = chunk_file.stem
        if book_id not in tags_map:
            continue
 
        with tags_map[book_id].open("r", encoding="utf-8") as f:
            tags_data = json.load(f)
 
        chunks = []
        with chunk_file.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        chunks.sort(key=lambda x: x["chunk_id"])
 
        total_chunks = len(chunks)
        if total_chunks == 0:
            chunks_analyze[book_id] = {}
            continue
 
        # ── Initialize stats ─────────────────────────────────────────────
        stats = {}
        for tag in TRACKED_TAGS:
            stats[tag] = {
                # Tag w całości mieści się w jednym chunku
                "inside": {"num": 0, "tag_ids": []},
                # Tag jest przecięty przez granicę chunka.
                # ZMIANA: tag_ids to zbiór unikalnych tag_id, żeby nie liczyć
                # tego samego tagu wielokrotnie (raz na każdy chunk, który go
                # dotyka).
                "splited": {"num": 0, "tag_ids": []},
                # NOWE: dla każdego pociętego tagu, przez ile chunków się
                # rozciąga. Pozwala ocenić, czy split jest "prawie dobry"
                # (2 chunki) czy katastrofalny (10 chunków).
                "split_chunk_spans": [],
            }
 
        stats["split_point"] = {
            # Chunk, którego start_index leży przy granicy otwierającego tagu
            "good_start": {"num": 0, "chunks_idx": []},
            # Chunk, którego end_index leży przy granicy zamykającego tagu
            "good_end": {"num": 0, "chunks_idx": []},
        }
 
        # ── Helpers ──────────────────────────────────────────────────────
 
        def gap_before(chunk_idx):
            if chunk_idx == 0:
                return None
            g_start = chunks[chunk_idx - 1]["end_index"] + 1
            g_end = chunks[chunk_idx]["start_index"] - 1
            if g_start > g_end:
                return None
            return (g_start, g_end)
 
        def gap_after(chunk_idx):
            if chunk_idx >= len(chunks) - 1:
                return None
            g_start = chunks[chunk_idx]["end_index"] + 1
            g_end = chunks[chunk_idx + 1]["start_index"] - 1
            if g_start > g_end:
                return None
            return (g_start, g_end)
 
        # ── Tag classification ────────────────────────────────────────────
        # Zbieramy najpierw dla każdego tag_id, w których chunkach się pojawia,
        # a dopiero potem klasyfikujemy. Dzięki temu:
        # 1. Unikamy wielokrotnego liczenia tego samego tagu
        # 2. Możemy policzyć split_chunk_span (przez ile chunków się rozciąga)
 
        tag_chunk_membership = defaultdict(list)  # tag_id -> [chunk_idx, ...]
 
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = chunk["start_index"]
            chunk_end = chunk["end_index"]
 
            for tag_id, tag_info in tags_data.items():
                if tag_info["tag_name"] not in TRACKED_TAGS:
                    continue
                tag_start = tag_info["tag_start"]
                tag_end = tag_info["tag_end"]
 
                if tag_start > chunk_end or tag_end < chunk_start:
                    continue  # brak overlapa
 
                tag_chunk_membership[tag_id].append(chunk_idx)
 
        # Teraz klasyfikuj każdy tag dokładnie raz
        for tag_id, chunk_indices in tag_chunk_membership.items():
            tag_info = tags_data[tag_id]
            tag_name = tag_info["tag_name"]
            tag_start = tag_info["tag_start"]
            tag_end = tag_info["tag_end"]
 
            if len(chunk_indices) == 1:
                # Tag mieści się w całości w jednym chunku
                chunk_idx = chunk_indices[0]
                chunk = chunks[chunk_idx]
                if tag_start >= chunk["start_index"] and tag_end <= chunk["end_index"]:
                    stats[tag_name]["inside"]["num"] += 1
                    stats[tag_name]["inside"]["tag_ids"].append({
                        "tag_id": tag_id,
                        "chunk_id": chunk["chunk_id"],
                    })
                else:
                    # Overlap, ale nie fully inside – np. tag leży na granicy
                    # jednego chunku (koniec tagu = koniec chunka, ale start
                    # tagu wypada w przerwie przed chunkiem)
                    stats[tag_name]["splited"]["num"] += 1
                    stats[tag_name]["splited"]["tag_ids"].append(tag_id)
                    stats[tag_name]["split_chunk_spans"].append(1)
            else:
                # Tag rozciąga się na wiele chunków – definitywnie splited
                stats[tag_name]["splited"]["num"] += 1
                stats[tag_name]["splited"]["tag_ids"].append(tag_id)
                stats[tag_name]["split_chunk_spans"].append(len(chunk_indices))
 
        # ── Per-tag derived stats ─────────────────────────────────────────
        for tag in TRACKED_TAGS:
            s = stats[tag]
            total_occurrences = s["inside"]["num"] + s["splited"]["num"]
            s["total"] = total_occurrences
            # Średnia rozpiętość splitów (ile chunków "zjada" jeden tag)
            spans = s["split_chunk_spans"]
            s["avg_split_span"] = (
                sum(spans) / len(spans) if spans else None
            )
            s["max_split_span"] = max(spans) if spans else None
 
        # ── Block Integrity Rate (BIR) ────────────────────────────────────
        # Procent tagów, które NIE są przecięte.
        # Liczymy tylko unikalne tagi (nie zdarzenia overlapa).
        # BIR = sum(inside[tag]) / sum(total[tag]) dla wszystkich TRACKED_TAGS
        total_inside = sum(stats[t]["inside"]["num"] for t in TRACKED_TAGS)
        total_all = sum(stats[t]["total"] for t in TRACKED_TAGS)
        stats["block_integrity_rate"] = (
            total_inside / total_all if total_all > 0 else None
        )
 
        # ── Split-point classification ────────────────────────────────────
        # Używamy chunk_idx zamiast chunk_id do wykluczania
        # pierwszego/ostatniego chunka – chunk_id może nie być sekwencyjne.
        total_boundaries = total_chunks - 1  # liczba wewnętrznych granic
 
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = chunk["start_index"]
            chunk_end = chunk["end_index"]
 
            gb = gap_before(chunk_idx)
            start_zone_lo = gb[0] if gb else chunk_start
            start_zone_hi = chunk_start
 
            ga = gap_after(chunk_idx)
            end_zone_lo = chunk_end
            end_zone_hi = ga[1] if ga else chunk_end
 
            good_start = False
            good_end = False
 
            for tag_id, tag_info in tags_data.items():
                if tag_info["tag_name"] not in TRACKED_TAGS:
                    continue
 
                t_start = tag_info["tag_start"]
                t_end = tag_info["tag_end"]
 
                # good_start: tag otwiera się przy początku tego chunka.
                # Pomijamy chunk_idx == 0 – pierwszy chunk nie ma lewej granicy.
                if not good_start and chunk_idx != 0:
                    if start_zone_lo <= t_start <= start_zone_hi:
                        good_start = True
 
                # good_end: tag zamyka się przy końcu tego chunka.
                # Pomijamy chunk_idx == total_chunks - 1 – ostatni chunk nie ma
                # prawej granicy.
                if not good_end and chunk_idx != total_chunks - 1:
                    if end_zone_lo <= t_end <= end_zone_hi:
                        good_end = True
 
                if good_start and good_end:
                    break
 
            if chunk_idx != 0 and good_start:
                stats["split_point"]["good_start"]["num"] += 1
                stats["split_point"]["good_start"]["chunks_idx"].append(chunk_idx)
 
            if chunk_idx != total_chunks - 1 and good_end:
                stats["split_point"]["good_end"]["num"] += 1
                stats["split_point"]["good_end"]["chunks_idx"].append(chunk_idx)
 
        # ── Boundary Preservation Rate (BPR) ─────────────────────────────
        # BPR = good_start / total_boundaries  i analogicznie dla end.
        # Dzielnik to total_boundaries (wewnętrzne granice), bo pierwszego
        # i ostatniego chunka nie analizujemy.
        sp = stats["split_point"]
        sp["bpr_start"] = (
            sp["good_start"]["num"] / total_boundaries
            if total_boundaries > 0 else None
        )
        sp["bpr_end"] = (
            sp["good_end"]["num"] / total_boundaries
            if total_boundaries > 0 else None
        )
 
        # ── Boundary Violation Count (BVC) ───────────────────────────────
        # Liczba granic między chunkami, gdzie ŻADNA strona nie jest
        # wyrównana do granicy tagu. Każda granica to para
        # (chunk[i].end, chunk[i+1].start). Używamy chunk_idx jako kluczy.
        good_end_set = set(sp["good_end"]["chunks_idx"])
        good_start_set = set(sp["good_start"]["chunks_idx"])
        bvc = 0
        for i in range(total_chunks - 1):
            # Lewa strona granicy: chunk i ma good_end
            # Prawa strona granicy: chunk i+1 ma good_start
            if i not in good_end_set and (i + 1) not in good_start_set:
                bvc += 1
        sp["boundary_violation_count"] = bvc
        sp["boundary_violation_rate"] = (
            bvc / total_boundaries if total_boundaries > 0 else None
        )
 
        # ── Coverage stats ────────────────────────────────────────────────
        # Chunki bez żadnego śledzonego tagu = "orphaned". Korzystamy
        # z tag_chunk_membership zbudowanego wcześniej: jeśli chunk_idx
        # nie pojawia się w żadnej liście wartości, chunk jest orphaned.
        chunks_with_tags = set()
        for tag_indices in tag_chunk_membership.values():
            chunks_with_tags.update(tag_indices)
 
        orphaned_chunks = sum(
            1 for i in range(total_chunks) if i not in chunks_with_tags
        )
        stats["coverage"] = {
            "orphaned_chunks": orphaned_chunks,
            "orphaned_chunk_rate": (
                orphaned_chunks / total_chunks if total_chunks > 0 else None
            ),
        }
 
        stats["total_chunks"] = total_chunks
        stats["total_boundaries"] = total_boundaries
        chunks_analyze[book_id] = stats
 
    total_time = time.perf_counter() - start_time
    return chunks_analyze, total_time
