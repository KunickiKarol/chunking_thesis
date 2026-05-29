import re


def extract_answer(raw: str) -> str:
    """
    Wyciąga tekst po ostatnim wystąpieniu:
    - "Answer:"
    - "Answers:"

    Obsługuje:
    - różne wielkości liter
    - bloki <think>...</think>
    - fallbacki dla pustych wyników
    """

    original_raw = raw

    # 1. Usuń bloki <think>...</think>
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # 2. Znajdź ostatnie wystąpienie "Answer:" lub "Answers:"
    matches = list(
        re.finditer(
            r"answers?\s*:\s*(.*)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    if matches:
        result = matches[-1].group(1).strip()

        # jeśli coś znaleziono i nie jest puste
        if result:
            return result

    # 3. Jeśli po usunięciu <think> coś zostało → zwróć wszystko po </think>
    after_think = re.split(
        r"</think>",
        original_raw,
        flags=re.IGNORECASE,
    )

    if len(after_think) > 0:
        result = after_think[-1].strip()
        if result:
            return result

    # 4. Ostateczny fallback
    return original_raw.strip()


def extract_verdict(raw: str) -> str:
    """
    Extracts final verdict from model output.

    Rules:
    - Take LAST occurrence of "Verdict:"
    - If value after it contains "incorrect" → return "incorrect"
    - If value contains "correct" → return "correct"
    - Case-insensitive
    - Otherwise → "incorrect"

    Handles:
    - <think>...</think> blocks
    """

    if not raw:
        return "incorrect"

    # 1. Remove <think> blocks
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 2. Find all verdict occurrences
    matches = list(
        re.finditer(
            r"verdict\s*:\s*(.*)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    if matches:
        last = matches[-1].group(1).strip().lower()

        # 3. Decision logic (STRICT PRIORITY)
        if "incorrect" in last:
            return "incorrect"
        if "correct" in last:
            return "correct"

        return "incorrect"

    # 4. Fallback: try after </think>
    split_think = re.split(
        r"</think>",
        raw,
        flags=re.IGNORECASE,
    )

    if split_think:
        tail = split_think[-1].strip().lower()
        if "incorrect" in tail:
            return "incorrect"
        if "correct" in tail:
            return "correct"

    # 5. Final fallback
    return "incorrect"
