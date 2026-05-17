from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    start_index: int
    end_index: int

def trim_bounds(s, start, end):
    lt = len(s) - len(s.lstrip())
    rt = len(s) - len(s.rstrip())
    return s.strip(), start + lt, end - rt -1