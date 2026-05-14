import re

def _extract_answer(raw: str) -> str:
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
    ).strip()

    if len(after_think) > 0:
        result = after_think[-1]
        if result:
            return result

    # 4. Ostateczny fallback
    return original_raw.strip()