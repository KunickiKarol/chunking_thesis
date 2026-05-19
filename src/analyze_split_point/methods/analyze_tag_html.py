import json
import time
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

        # ── Initialize stats ────────────────────────────────────────────────
        stats = {}
        for tag in TRACKED_TAGS:
            stats[tag] = {
                "inside":  {"num": 0, "tag_ids": []},  # (tag_id, chunk_id)
                "splited": {"num": 0, "tag_ids": []},  # (tag_id, chunk_id)
            }
        stats["split_point"] = {
            # chunk whose start_index aligns with a tag boundary
            "good_start": {"num": 0, "chunks_id": []},
            # chunk whose end_index aligns with a tag boundary
            "good_end":   {"num": 0, "chunks_id": []},
        }

        # ── Helpers ─────────────────────────────────────────────────────────

        def gap_before(chunk_idx):
            """
            Return (gap_start, gap_end) — the whitespace gap immediately before
            chunks[chunk_idx], or None if chunk_idx == 0.
            Both bounds are inclusive character positions.
            end_index is inclusive, so the gap is:
              [chunks[chunk_idx-1].end_index + 1 … chunks[chunk_idx].start_index - 1]
            """
            if chunk_idx == 0:
                return None
            g_start = chunks[chunk_idx - 1]["end_index"] + 1
            g_end   = chunks[chunk_idx]["start_index"] - 1
            if g_start > g_end:   # chunks are adjacent, no gap
                return None
            return (g_start, g_end)

        def gap_after(chunk_idx):
            """
            Return (gap_start, gap_end) — the whitespace gap immediately after
            chunks[chunk_idx], or None if chunk_idx is the last chunk.
            Both bounds are inclusive character positions.
              [chunks[chunk_idx].end_index + 1 … chunks[chunk_idx+1].start_index - 1]
            """
            if chunk_idx >= len(chunks) - 1:
                return None
            g_start = chunks[chunk_idx]["end_index"] + 1
            g_end   = chunks[chunk_idx + 1]["start_index"] - 1
            if g_start > g_end:
                return None
            return (g_start, g_end)

        def pos_touches_chunk_start(pos, chunk_idx):
            """
            True if `pos` is at or in the gap just before chunks[chunk_idx],
            i.e. it is a valid 'start of chunk' position.
            Valid range: [gap_before_start … chunk_start_index] (inclusive).
            """
            chunk_start = chunks[chunk_idx]["start_index"]
            if pos == chunk_start:
                return True
            gap = gap_before(chunk_idx)
            if gap and gap[0] <= pos <= gap[1]:
                return True
            return False

        def pos_touches_chunk_end(pos, chunk_idx):
            """
            True if `pos` is at or in the gap just after chunks[chunk_idx],
            i.e. it is a valid 'end of chunk' position.
            Valid range: [chunk_end_index … gap_after_end] (inclusive).
            """
            chunk_end = chunks[chunk_idx]["end_index"]
            if pos == chunk_end:
                return True
            gap = gap_after(chunk_idx)
            if gap and gap[0] <= pos <= gap[1]:
                return True
            return False

        # ── Tag classification (inside / splited) ───────────────────────────
        # For every tracked tag, check it against every chunk it overlaps.
        # This correctly handles overlapping chunks: the same tag may appear in
        # multiple (chunk, category) pairs.

        for tag_id, tag_info in tags_data.items():
            tag_name  = tag_info["tag_name"]
            tag_start = tag_info["tag_start"]
            tag_end   = tag_info["tag_end"]

            if tag_name not in stats:
                continue

            for chunk in chunks:
                chunk_id    = chunk["chunk_id"]
                chunk_start = chunk["start_index"]
                chunk_end   = chunk["end_index"]

                # Does this tag overlap this chunk at all?
                # Overlap: tag_start <= chunk_end  AND  tag_end >= chunk_start
                if tag_start > chunk_end or tag_end < chunk_start:
                    continue  # no overlap with this chunk

                # INSIDE: tag is fully contained within the chunk
                if tag_start >= chunk_start and tag_end <= chunk_end:
                    stats[tag_name]["inside"]["num"] += 1
                    stats[tag_name]["inside"]["tag_ids"].append({'tag_id': tag_id, 'chunk_id': chunk_id})
                else:
                    # SPLITED: tag overlaps the chunk but bleeds outside it
                    stats[tag_name]["splited"]["num"] += 1
                    stats[tag_name]["splited"]["tag_ids"].append({'tag_id': tag_id, 'chunk_id': chunk_id})

        # ── Split-point classification (good_start / good_end) ──────────────
        # For each chunk, check whether any tracked tag begins at its start
        # boundary (good_start) or ends at its end boundary (good_end).
        # The gap adjacent to the boundary counts as part of that boundary.
        total_chunks = len(chunks)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id    = chunk["chunk_id"]
            chunk_start = chunk["start_index"]
            chunk_end   = chunk["end_index"]

            # Effective start zone: [gap_before_start … chunk_start] (inclusive)
            gb = gap_before(chunk_idx)
            start_zone_lo = gb[0] if gb else chunk_start
            start_zone_hi = chunk_start

            # Effective end zone: [chunk_end … gap_after_end] (inclusive)
            ga = gap_after(chunk_idx)
            end_zone_lo = chunk_end
            end_zone_hi = ga[1] if ga else chunk_end

            good_start_found = False
            good_end_found   = False

            for tag_id, tag_info in tags_data.items():
                if tag_info["tag_name"] not in TRACKED_TAGS:
                    continue

                t_start = tag_info["tag_start"]
                t_end   = tag_info["tag_end"]

                # good_start: a tag opens exactly at (or in the gap before) this chunk
                if not good_start_found:
                    if start_zone_lo <= t_start <= start_zone_hi:
                        if chunk_id != 0:
                            stats["split_point"]["good_start"]["num"] += 1
                            stats["split_point"]["good_start"]["chunks_id"].append(chunk_id)
                            good_start_found = True

                # good_end: a tag closes exactly at (or in the gap after) this chunk
                if not good_end_found:
                    if end_zone_lo <= t_end <= end_zone_hi:
                        if chunk_id != len(chunks) - 1:
                            stats["split_point"]["good_end"]["num"] += 1
                            stats["split_point"]["good_end"]["chunks_id"].append(chunk_id)
                            good_end_found = True

                if good_start_found and good_end_found:
                    break  # no need to check more tags for this chunk
        stats["total_chunks_analyzed"] = (total_chunks - 1) or 1
        chunks_analyze[book_id] = stats
    total_time = time.perf_counter() - start_time
    return chunks_analyze, total_time
