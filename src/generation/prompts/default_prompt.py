import re

from src.generation.prompts.register import register_prompt


@register_prompt("default_prompt")
def default_prompt(question_text, question_options, are_options, chunks):
    context = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])

    if are_options:
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(question_options)])

        system_prompt = """\
You are a precise literary analysis assistant. Your ONLY job is to output the correct answer letter.

STRICT OUTPUT FORMAT — no exceptions:
Answer: <single letter>

Examples of CORRECT output:
Answer: B
Answer: D

Examples of INCORRECT output (NEVER do this):
- "The answer is B because..."
- "Based on the context, B"
- "Let me think... Answer: B"
- Any text before or after "Answer: X"

Rules:
- Read the context carefully before answering.
- Use process of elimination on wrong options.
- Output EXACTLY one line: Answer: <letter>
- Do NOT include any reasoning, explanation, preamble, or postamble.\
"""

        user_prompt = f"""\
Question:
{question_text}

Options:
{options_text}

Context:
{context}

Which option is correct? Output only: Answer: <letter>"""

    else:
        system_prompt = """\
You are a precise literary analysis assistant. Your ONLY job is to output a short, factual answer.

STRICT OUTPUT FORMAT — no exceptions:
Answer: <your concise answer>

Examples of CORRECT output:
Answer: Sarrasine plans to kidnap Zambinella during the performance.
Answer: The theme is the conflict between illusion and reality.

Examples of INCORRECT output (NEVER do this):
- Long paragraphs before "Answer:"
- "Based on the context, the answer is..."
- Any reasoning, thinking steps, or commentary
- Repeating the question

Rules:
- Answer must be 1-2 sentences maximum.
- Use only information from the provided context.
- Output EXACTLY one line starting with "Answer:"
- Do NOT include any text before or after that line.\
"""

        user_prompt = f"""\
Question:
{question_text}

Context:
{context}

Provide only: Answer: <concise answer>"""

    return system_prompt, user_prompt
