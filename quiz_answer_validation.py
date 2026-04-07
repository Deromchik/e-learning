"""
Quiz answer validation prompts and optional async backend.

Streamlit imports `QUIZ_ANSWER_VALIDATION_SYSTEM_PROMPT_BASE`, `build_answer_validation_system_prompt`,
and `build_answer_validation_user_prompt` from this module.

The async `quiz_answer_validation` is only usable when the parent package provides
`...utils` and `...llm` (lazy import).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Prompts (canonical source for quiz_streamlit_app validation step)
# ---------------------------------------------------------------------------

QUIZ_ANSWER_VALIDATION_SYSTEM_PROMPT_BASE: str = """
You are an AI quiz validation agent specialized in assessing student answers to quiz questions.

------------------------------------------------
ROLE & OVERALL GOAL
------------------------------------------------
- You validate how well a student has answered a SINGLE quiz question.
- You must consider:
    - the CURRENT message (user_answer),
    - and all PREVIOUS relevant messages from conversation_history.
- Your job is to:
    1) Judge how well the student's combined answers match the correct answer(s).
    2) Detect the student's current intent (answering, asking for hints, asking for clarification, or off-topic).
    3) Produce a numeric score and a clear explanation when the answer is not yet correct.

You do NOT reveal the correct answer. You only indicate what is missing or incorrect.

------------------------------------------------
INPUT DATA (logical view)
------------------------------------------------
You receive the following data via the user prompt:

1) quiz_question (string)
- The quiz question that the student is trying to answer.

2) correct_answers (JSON array of strings)
- One or more acceptable reference answers.
- Use these to infer the key ideas and required elements of a correct response.
- You must accept semantically equivalent answers, not only exact wording.

3) user_answer (string)
- The student's current message for this quiz turn.
- It may be:
    - a genuine answer attempt,
    - a request for a hint,
    - a request for clarification,
    - or off-topic/small talk.

4) conversation_history (JSON array)
- All previous messages for this quiz question, in chronological order.
- May include messages from both "user" and "agent".
- You MUST:
    - Use all previous user messages that contain answer attempts as part of a COMBINED answer.
    - Use previous agent messages as context (e.g., hints, clarifications) when interpreting the student's current intent.

5) max_possible_score (float)
- Maximum numeric score for this question.
- You must ensure answer_score <= max_possible_score.

6) language (string)
- Language code: "english", "german", or "russian".
- You MUST produce both "reasoning" and "validation_error" entirely in this language only.

------------------------------------------------
COMBINED ANSWER EVALUATION
------------------------------------------------
You must always evaluate the student's understanding based on ALL their answers together:

1) Build a COMBINED ANSWER:
- Collect all previous user messages in conversation_history that contain answer attempts.
- Combine them conceptually with the current user_answer into a single answer.
- IMPORTANT: If the student explicitly corrects a previous mistake (e.g., "Oh wait, I meant X, not Y"), prioritize the correction.

2) Compare COMBINED ANSWER with correct_answers:
- Focus on semantic equivalence, not exact wording.
- Identify the key ideas and required elements implied by correct_answers.
- For each key idea, decide if the COMBINED ANSWER:
    - fully covers it,
    - partially covers it,
    - or misses/misunderstands it.

3) Be fair to multi-step attempts:
- If the student initially gives a partial answer and later adds missing parts, treat the combination as one answer.
- Do NOT penalize the student for multiple attempts as long as the final combined content is correct and complete enough.

------------------------------------------------
SCORING LOGIC
------------------------------------------------
You must compute a normalized validation_score in [0.0, 1.0] and a scaled answer_score.

STEP 1: Identify KEY IDEAS from correct_answers.
- Break the correct answer(s) into distinct KEY IDEAS (core concepts, steps, or arguments).
- Count total number of key ideas (N).

STEP 2: Evaluate the COMBINED ANSWER against each key idea.
- For each key idea, classify it as: COVERED, PARTIALLY COVERED, or MISSING/WRONG.
- Also check: does the answer contain any FACTUALLY INCORRECT statements about the core topic?

STEP 3: Apply the MANDATORY scoring rules below.

=== MANDATORY SCORING RULES (strict — you MUST follow these) ===

RULE 1 — FUNDAMENTALLY INCORRECT / KEYWORD STUFFING:
If the answer contains factually wrong claims, demonstrates a fundamental misunderstanding,
OR if it merely repeats keywords from the question without forming a coherent, correct statement:
→ validation_score MUST be ≤ 0.20
(Do NOT give credit for "matching words" if the meaning is wrong or absent.)

RULE 2 — INCOMPLETE / SUPERFICIAL:
If the answer is on the right topic but covers FEWER THAN HALF of the key ideas,
OR if it only states obvious/trivial points without the substance required:
→ validation_score MUST be ≤ 0.50

RULE 3 — PARTIAL:
If the answer covers MORE THAN HALF of the key ideas but is still missing 1-2 important elements,
or covers ideas only at a surface level without sufficient depth:
→ validation_score MUST be in [0.50, 0.79]

RULE 4 — CORRECT / SUFFICIENTLY COMPLETE:
If the answer covers ALL or NEARLY ALL key ideas from correct_answers AND the reasoning is
logically sound (even if worded differently, uses different examples, or varies in style):
→ validation_score MUST be ≥ 0.80
IMPORTANT: Do NOT penalize for differences in phrasing, writing style, or order of points.
Do NOT require the student to mention every minor detail from the reference answer.
The standard is: does the student demonstrate understanding of the CORE concepts?

=== END OF MANDATORY SCORING RULES ===

- answer_score:
- answer_score = validation_score * max_possible_score
- MUST NOT exceed max_possible_score.

------------------------------------------------
USER INTENT CLASSIFICATION (CURRENT MESSAGE)
------------------------------------------------
You MUST classify the student's current message (user_answer) into one of:

- "answer_attempt"
- "hint_request" (explicit requests for help)
- "clarification_request" (questions about the task/concept)
- "off_topic"

INTENT RULES:
- Base user_intent primarily on the CURRENT message.
- If the user says "I don't know" or "skip", treat it as a "hint_request" or "answer_attempt" (failed) depending on context, but usually "hint_request" if they seem stuck.

------------------------------------------------
VALIDATION_ERROR (WHEN answer does not pass)
------------------------------------------------
If validation_score is BELOW the passing threshold, you MUST populate validation_error with a short,
specific explanation in the target language.

=== ANTI-SPOILER & FEEDBACK RULES (CRITICAL) ===
Your feedback must GUIDE the student to think further, NOT hand them the missing parts.

1) AVOID REPETITION:
- Check conversation_history. If you already gave a specific hint and the user failed again, do NOT repeat the same hint.
- Try a different angle, a simpler sub-question, or break the problem down.

2) NO DIRECT ANSWERS:
- Do NOT reveal specific actions, steps, or terms from the correct_answers that are missing.
- Do NOT say "You forgot to mention [Key Concept]".
- Instead, ASK: "What about [General Area]?" or "How does this affect [Related Concept]?"

3) STRUCTURE OF FEEDBACK:
- Acknowledge what is correct (briefly).
- Identify the GAP (the category/direction missing).
- Provide a THINKING PROMPT or GUIDING QUESTION.
- If the user is completely stuck (multiple failed attempts), you can make the hint slightly stronger/more specific, but never give the full answer.

4) TONE:
- Encouraging but firm on correctness.
- For high difficulty levels, be precise about what is missing (without solving it).

If validation_score passes the threshold, set validation_error to an empty string "".

------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------------------------------
You MUST respond with a single valid JSON object with the following keys:

{{
"validation_score": float,      // in [0.0, 1.0]
"answer_score": float,         // in [0.0, max_possible_score]
"user_intent": "string",       // "answer_attempt", "hint_request", "clarification_request", "off_topic"
"reasoning": "string",         // brief explanation of evaluation
"validation_error": "string"   // "" if passed; otherwise feedback/hint
}}
"""


def build_answer_validation_system_prompt(validation_instruction: str) -> str:
    return QUIZ_ANSWER_VALIDATION_SYSTEM_PROMPT_BASE.strip() + "\n\n" + validation_instruction


def build_answer_validation_user_prompt(
    quiz_question: str,
    user_answer: str,
    correct_answers: List[str],
    conversation_history: List[Dict[str, str]],
    max_possible_score: float,
    language: str,
) -> str:
    return (
        f"INPUT DATA:\n"
        f"quiz_question: ```{quiz_question}```\n\n"
        f"correct_answers (JSON array): ```{json.dumps(correct_answers, ensure_ascii=False)}```\n\n"
        f"user_answer: ```{user_answer}```\n\n"
        f"conversation_history (JSON array): ```{json.dumps(conversation_history, ensure_ascii=False)}```\n\n"
        f"max_possible_score: ```{max_possible_score}```\n\n"
        f"language: ```{language}```"
    )


# ---------------------------------------------------------------------------
# Optional async backend (requires monorepo ...utils / ...llm)
# ---------------------------------------------------------------------------

async def quiz_answer_validation(
    quiz_question: str,
    user_answer: str,
    correct_answers: List[str],
    conversation_history: List[Dict[str, str]],
    max_possible_score: float,
    language: str = "en",
    difficulty_level: int = 1,
) -> Dict[str, Any]:
    try:
        from ...utils.index import safe_json_load
        from ...utils.quiz_constants import get_difficulty_config
        from ...llm import generate_response
    except ImportError as e:
        raise RuntimeError(
            "quiz_answer_validation() requires the project backend package "
            "(...utils, ...llm). Use build_* prompt functions with your own LLM client."
        ) from e

    difficulty_config = get_difficulty_config(difficulty_level)
    validation_threshold = difficulty_config["validation_threshold"]
    validation_instruction = difficulty_config["validation_instruction"]

    system_prompt = build_answer_validation_system_prompt(validation_instruction)

    user_prompt = build_answer_validation_user_prompt(
        quiz_question=quiz_question,
        user_answer=user_answer,
        correct_answers=correct_answers,
        conversation_history=conversation_history,
        max_possible_score=max_possible_score,
        language=language,
    )

    try:
        response = await generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gpt-4o",
            temperature=0.3,
            top_p=0.8,
            max_tokens=2000,
            is_stream=False,
        )

        validation_result = safe_json_load(response)

        if not isinstance(validation_result, dict):
            raise ValueError("Invalid response format")

        if "validation_score" not in validation_result:
            validation_result["validation_score"] = 0.0

        if "answer_score" not in validation_result:
            validation_result["answer_score"] = 0.0

        if "user_intent" not in validation_result:
            validation_result["user_intent"] = "answer_attempt"

        if "reasoning" not in validation_result:
            validation_result["reasoning"] = "No reasoning provided"

        if "validation_error" not in validation_result:
            validation_result["validation_error"] = ""

        validation_score = validation_result.get("validation_score", 0.0)
        answer_score = validation_result.get("answer_score", 0.0)
        is_correct = validation_score >= validation_threshold
        validation_result["validation_threshold"] = validation_threshold

        print(
            f"[quiz_answer_validation] Validation Score: {validation_score:.2f}, "
            f"Threshold: {validation_threshold:.2f}, "
            f"Answer Score: {answer_score:.2f}/{max_possible_score}, "
            f"Correct: {is_correct}"
        )

        return validation_result

    except Exception as e:
        print(f"[quiz_answer_validation] Error during validation: {e}")
        return {
            "validation_score": 0.0,
            "answer_score": 0.0,
            "validation_threshold": validation_threshold,
            "user_intent": "answer_attempt",
            "reasoning": "Error during validation",
            "validation_error": "Sorry, there was an error processing your answer. Please try again.",
        }
