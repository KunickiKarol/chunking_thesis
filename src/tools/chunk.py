from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    start_index: int
    end_index: int
