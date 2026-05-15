from src.evaluate_generator.prompts.register import register_prompt

@register_prompt("default_prompt")
def default_prompt(
    question_text,
    gold_answers_text,
    llm_answer
):

    system_prompt = """\
You are a highly strict and reliable evaluation engine for answer correctness.

Your role is to act as an LLM-as-a-judge that compares a candidate answer against a set of reference (gold) answers and decides whether it is correct.

You do NOT solve the question. You only evaluate semantic equivalence.

========================
EVALUATION PRINCIPLES
========================

1. Semantic equivalence is required:
   - The candidate answer is correct if it expresses the same meaning as ANY one gold answer.
   - Paraphrases, reordering, and minor syntactic differences are acceptable.

2. No guessing:
   - If equivalence is uncertain → mark INCORRECT.

3. Ignore style:
   - Ignore grammar, formatting, verbosity, and tone.

4. Focus ONLY on meaning:
   - Do not be influenced by length, confidence, or persuasion.

========================
OUTPUT CONSTRAINTS (HARD RULE)
========================

Return ONLY one line:

Verdict: correct
OR
Verdict: incorrect

No explanations.
No additional text.
No punctuation variations.
No extra lines.

Any deviation is invalid.
"""

    user_prompt = f"""\
Question:
{question_text}

Gold answers (reference set, ANY one is sufficient):
{gold_answers_text}

Candidate answer:
{llm_answer}

Task:
Compare the candidate answer with the gold answers and decide if it is semantically equivalent to at least one of them.

Return only:
Verdict: correct
or
Verdict: incorrect
"""

    return system_prompt, user_prompt