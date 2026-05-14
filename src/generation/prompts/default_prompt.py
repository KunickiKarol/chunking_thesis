from src.generation.prompts.register import register_prompt


@register_prompt("default_prompt")
def default_prompt(question_text, question_options, are_options, chunks):
    context = "\n".join([f"- {chunk}" for chunk in chunks])

    if are_options:
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(question_options)])

        system_prompt = f"""
You are a world-class literature expert and exam-solving assistant.

Your task is to answer literature-related questions with extremely high accuracy.
Use careful reasoning, contextual analysis, elimination techniques, semantic matching,
and educational best practices.

Rules:
- Be concise and precise.
- Prefer the most probable answer.
- If multiple-choice options are provided, answer ONLY with the correct letter (A, B, C, D, ...).
- Do not explain unless explicitly required.
- Use the provided context heavily.
- Think step-by-step internally but output only the final answer.
- Output must have the format: "Answer: X" where X is the letter of the correct option.
"""

        user_prompt = f"""
Question:
{question_text}

Options:
{options_text}

Context:
{context}

Choose the single best answer.
Return ONLY the letter (A, B, C, D, ...).
"""

    else:
        system_prompt = f"""
You are a world-class literature expert and academic assistant.

Your task is to answer literature-related questions accurately,
clearly, and concisely.

Rules:
- Keep answers short and concrete.
- Use the provided context heavily.
- Avoid unnecessary explanations.
- Focus on factual correctness and literary interpretation quality.
- If the question is ambiguous, provide the most likely interpretation and answer that.
- Output must have the format: "Answer: [your answer here]".
"""

        user_prompt = f"""
Question:
{question_text}

Context:
{context}

Provide a short and precise answer.
"""

    return system_prompt, user_prompt
