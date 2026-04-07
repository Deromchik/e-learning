#!/usr/bin/env python3
"""
Streamlit Quiz App
Combines prompt_review_runner_2-style question generation with the quiz pipeline
prompts from quiz_answer_validation.py, quiz_question_rephraser.py,
quiz_followup_question.py, and quiz_completion_message.py (imported as the
canonical prompt source). All API calls are logged; download JSON or a TXT prompt stream
from the sidebar. Quiz attempt limits apply only to intents other than clarification_request
and off_topic.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

# User intents that are not graded quiz answers — do not consume max_attempts.
_INTENTS_EXEMPT_FROM_ATTEMPT_QUOTA: frozenset[str] = frozenset(
    {"clarification_request", "off_topic"}
)


def _normalize_user_intent_for_quota(raw: object) -> str:
    s = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return s

from quiz_answer_validation import (
    build_answer_validation_system_prompt,
    build_answer_validation_user_prompt,
)
from quiz_completion_message import (
    QUIZ_COMPLETION_SYSTEM_PROMPT,
    build_completion_user_prompt,
)
from quiz_followup_question import (
    QUIZ_FOLLOWUP_SYSTEM_PROMPT,
    build_followup_user_prompt,
)
from quiz_question_rephraser import (
    QUIZ_REPHRASER_SYSTEM_PROMPT,
    build_rephraser_user_prompt,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OPENROUTER_MAX_TOKENS: int = 10000

LANGUAGE_MAP: dict[str, str] = {
    "en": "English", "ru": "Russian", "pl": "Polish", "uk": "Ukrainian",
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "cs": "Czech", "sk": "Slovak",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian",
    "sr": "Serbian", "sl": "Slovenian", "et": "Estonian", "lv": "Latvian",
    "lt": "Lithuanian", "fi": "Finnish", "sv": "Swedish", "no": "Norwegian",
    "da": "Danish", "is": "Icelandic", "ga": "Irish", "mt": "Maltese",
    "el": "Greek", "tr": "Turkish", "ar": "Arabic", "he": "Hebrew",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
    "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
    "tl": "Filipino", "bn": "Bengali", "fa": "Persian",
}


def get_language_name(code: str) -> str:
    return LANGUAGE_MAP.get(str(code).lower(), str(code).upper())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# DIFFICULTY LEVELS
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS: dict[int, dict] = {
    1: {
        "label": "1 — Beginner",
        "validation_threshold": 0.50,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 1 (BEGINNER) ===\n"
            "Generate VERY EASY questions suitable for absolute beginners.\n"
            "- Questions should test basic terminology recall and simple definitions.\n"
            "- Answers should be single words or very short phrases (1-2 words).\n"
            "- Avoid any questions requiring understanding of relationships between concepts.\n"
            "- Focus on the most fundamental, surface-level facts from the materials.\n"
            "- Each question should have a clear, unambiguous single correct answer.\n"
            "- Provide 3-5 acceptable answer variants (synonyms, abbreviations, alternate forms).\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 1 (BEGINNER) ===\n"
            "Generate VERY EASY reasoning questions suitable for absolute beginners.\n"
            "- Scenarios should be simple and straightforward with an obvious course of action.\n"
            "- Only one clear consideration or step is needed to solve the scenario.\n"
            "- Use everyday, relatable situations that directly map to a single concept from the lesson.\n"
            "- The expected reasoning should be short and simple (2-3 sentences).\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 1 — Beginner):\n"
            "The quiz difficulty is set to BEGINNER level. Be VERY lenient when scoring.\n"
            "- Accept answers that are approximately correct or in the right direction.\n"
            "- If the student demonstrates they understand the general area/topic, give generous credit.\n"
            "- Synonyms, related terms, and imprecise but directionally correct answers should score highly.\n"
            "- A validation_score >= 0.50 should be considered passing for this difficulty level.\n"
            "- However, do NOT give credit for random keywords if the answer makes no sense.\n"
            "- IMPORTANT: At beginner level, covering even 1-2 core ideas from the reference answer\n"
            "  in a generally correct way is enough for a passing score.\n"
        ),
    },
    2: {
        "label": "2 — Elementary",
        "validation_threshold": 0.60,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 2 (ELEMENTARY) ===\n"
            "Generate EASY questions suitable for learners with basic knowledge.\n"
            "- Questions should test simple understanding and basic recall.\n"
            "- Answers should be short (1-3 words).\n"
            "- Questions may test simple relationships between two concepts.\n"
            "- Stick to the most important, clearly stated facts from the materials.\n"
            "- Provide 2-4 acceptable answer variants.\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 2 (ELEMENTARY) ===\n"
            "Generate EASY reasoning questions for learners with basic knowledge.\n"
            "- Scenarios should be straightforward with a clear recommended action.\n"
            "- Require consideration of 1-2 factors from the lesson.\n"
            "- Situations should be practical and easy to relate to.\n"
            "- The expected reasoning should involve 2-4 simple steps.\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 2 — Elementary):\n"
            "The quiz difficulty is set to ELEMENTARY level. Be lenient when scoring.\n"
            "- Accept answers that capture the main idea even if details are missing.\n"
            "- Related terms and partially correct answers should receive partial credit.\n"
            "- A validation_score >= 0.60 should be considered passing for this difficulty level.\n"
            "- Penalize only clearly incorrect or unrelated answers (score < 0.30).\n"
            "- If the answer covers the main idea correctly, it should pass even without full detail.\n"
        ),
    },
    3: {
        "label": "3 — Intermediate",
        "validation_threshold": 0.70,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 3 (INTERMEDIATE) ===\n"
            "Generate MODERATE difficulty questions for learners with solid foundational knowledge.\n"
            "- Questions should test understanding of concepts and their relationships.\n"
            "- Answers should be concise (1-3 words) but require understanding, not just memorization.\n"
            "- Include questions about how concepts connect or differ from each other.\n"
            "- Provide 2-4 acceptable answer variants.\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 3 (INTERMEDIATE) ===\n"
            "Generate MODERATE reasoning questions for learners with solid knowledge.\n"
            "- Scenarios should involve multiple considerations and trade-offs.\n"
            "- Require applying knowledge from the lesson to realistic situations.\n"
            "- The expected reasoning should involve 3-5 steps with clear logic.\n"
            "- Include scenarios where more than one approach is possible but one is clearly better.\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 3 — Intermediate):\n"
            "The quiz difficulty is set to INTERMEDIATE level. Apply balanced scoring.\n"
            "- Answers should demonstrate understanding of the concept, not just recall.\n"
            "- Accept semantically equivalent answers with minor imprecisions.\n"
            "- A validation_score >= 0.70 should be considered passing for this difficulty level.\n"
            "- Partial credit (0.50-0.69) for answers that show understanding but miss important nuances.\n"
            "- If the student covers all core ideas correctly, give a passing score even if depth varies.\n"
        ),
    },
    4: {
        "label": "4 — Advanced",
        "validation_threshold": 0.80,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 4 (ADVANCED) ===\n"
            "Generate CHALLENGING questions for learners with strong knowledge.\n"
            "- Questions should test deeper understanding, analysis, and application.\n"
            "- Include questions about edge cases, exceptions, or subtle distinctions.\n"
            "- Require precise answers that demonstrate thorough understanding.\n"
            "- CRITICAL: Base the complexity ONLY on the provided materials. Do NOT invent concepts, frameworks, or facts that are not explicitly present in the text.\n"
            "- Provide 2-3 acceptable answer variants (less tolerance for imprecision).\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 4 (ADVANCED) ===\n"
            "Generate CHALLENGING reasoning questions for knowledgeable learners.\n"
            "- Scenarios should involve complex trade-offs and multiple interacting factors.\n"
            "- Require synthesis of several concepts from the lesson.\n"
            "- The expected reasoning should be thorough and well-structured.\n"
            "- Provide a structured answer (approx. 4-6 sentences) explaining the trade-offs and the rationale.\n"
            "- CRITICAL: Base the complexity ONLY on the provided materials. Do NOT invent concepts, frameworks, or facts that are not explicitly present in the text.\n"
            "- Include scenarios with non-obvious solutions that require critical analysis.\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 4 — Advanced):\n"
            "The quiz difficulty is set to ADVANCED level.\n"
            "- Answers must demonstrate clear understanding and cover the core concepts.\n"
            "- Semantically equivalent answers are accepted, but vague or overly general answers should score lower.\n"
            "- A validation_score >= 0.80 should be considered passing for this difficulty level.\n"
            "- CRITICAL: If the answer addresses ALL CORE ideas from the reference answer and the reasoning is\n"
            "  logically sound, you MUST give validation_score >= 0.80. Do NOT lower the score because the student\n"
            "  did not mention edge cases, ancillary details, or supplementary examples that go BEYOND the core ideas.\n"
            "- Strictness applies to INCORRECT or MISLEADING information, NOT to stylistic differences or minor depth gaps.\n"
            "- Do NOT be pedantic. If the student captures the ESSENCE of the complex concept, pass it.\n"
        ),
    },
    5: {
        "label": "5 — Expert",
        "validation_threshold": 0.88,
        "question_generation_instruction": (
            "=== DIFFICULTY LEVEL: 5 (EXPERT) ===\n"
            "Generate VERY DIFFICULT questions for expert-level learners.\n"
            "- Questions should test synthesis, critical evaluation, and nuanced understanding.\n"
            "- Include questions about implications, limitations, and advanced applications.\n"
            "- Require highly precise, technically accurate answers.\n"
            "- CRITICAL: Base the complexity ONLY on the provided materials. Do NOT invent concepts, frameworks, or facts that are not explicitly present in the text.\n"
            "- Provide only 1-2 acceptable answer variants (very strict matching).\n"
            "- Questions may combine concepts from current and previous lessons.\n\n"
        ),
        "reasoning_generation_instruction": (
            "=== DIFFICULTY LEVEL: 5 (EXPERT) ===\n"
            "Generate VERY DIFFICULT reasoning questions for expert learners.\n"
            "- Scenarios should be highly complex with multiple competing priorities.\n"
            "- Require deep synthesis of concepts from across all provided materials.\n"
            "- The expected reasoning should demonstrate expert-level analysis and judgment.\n"
            "- Include scenarios with ambiguity where the quality of reasoning matters most.\n"
            "- Provide a structured answer (approx. 4-6 sentences) explaining the trade-offs and the rationale.\n"
            "- CRITICAL: Base the complexity ONLY on the provided materials. Do NOT invent concepts, frameworks, or facts that are not explicitly present in the text.\n"
            "- Expect thorough consideration of consequences and edge cases.\n\n"
        ),
        "validation_instruction": (
            "DIFFICULTY ADJUSTMENT (Level 5 — Expert):\n"
            "The quiz difficulty is set to EXPERT level.\n"
            "- Answers must be precise, technically correct, and cover the core concepts thoroughly.\n"
            "- A validation_score >= 0.88 should be considered passing for this difficulty level.\n"
            "- Vague, incomplete, or imprecise answers should score significantly lower.\n"
            "- CRITICAL: Even at expert level, if the student addresses ALL MAIN concepts from the reference answer\n"
            "  correctly and demonstrates sound, well-structured reasoning, you MUST give validation_score >= 0.88.\n"
            "  Do NOT penalize for omitting minor supplementary details, edge cases not central to the question,\n"
            "  or for using different (but accurate) terminology.\n"
            "- Penalize ONLY for: (a) factually incorrect statements, (b) missing CORE elements that are essential\n"
            "  to the answer, (c) flawed reasoning logic, (d) fundamental misunderstanding of the topic.\n"
            "- Do NOT invent extra requirements beyond what is in the reference answer to fail the student.\n"
        ),
    },
}


def get_difficulty_config(level: int) -> dict:
    return DIFFICULTY_LEVELS.get(level, DIFFICULTY_LEVELS[3])


# ---------------------------------------------------------------------------
# PROMPT TEMPLATES (from prompt_review_runner_2.py)
# ---------------------------------------------------------------------------

PROMPT_SYSTEM_TEMPLATE: str = (
    "You are an expert assessment designer and pedagogy specialist. "
    "Your output will be used by a downstream LLM to grade learners across multiple scoring dimensions. "
    "Always write in clear, concise, natural {language_name} (language code: {course_language}). "
    "Favor measurable objectives, discriminative criteria, and observable evidence. Avoid fluff. "
)

PROMPT_GIFTQUIZ_QUESTIONS_TEMPLATE: str = (
    "You are an assessment designer. Generate SHORT ANSWER questions in {language_name} (language code: {course_language}) in strict JSON.\n"
    "Prioritize the CURRENT LESSON materials, use previous lesson summaries only if helpful.\n"
    "IMPORTANT: All questions and answers must be written in {language_name}.\n\n"
    "{difficulty_block}"
    "Materials (may be partial):\n"
    "Gift (raw, may include questions):\n{gift_text}\n\n"
    "{extracted_block}"
    "{other_block}"
    "{prev_block}"
    "=== QUESTION GENERATION GUIDELINES ===\n\n"
    "1. DIVERSITY (CRITICAL): Each question MUST test a DIFFERENT concept or term.\n"
    "   - Do NOT create multiple questions with the same answer.\n"
    "   - Cover the full range of key concepts from the materials.\n"
    "   - If existing questions cover a concept, create NEW questions about OTHER concepts.\n\n"
    "2. ANSWERS: Provide 2-4 acceptable answer variants (synonyms, abbreviations, alternate phrasings).\n"
    "   - Include both singular and plural forms if applicable.\n"
    "   - Include common abbreviations (e.g., 'AI', 'Artificial Intelligence').\n"
    "   - Answers should be SHORT (1-3 words typically).\n\n"
    "3. SCORE WEIGHTING (1-5 points per question):\n"
    "   - 1 point: Basic terminology recall (simple definitions)\n"
    "   - 2 points: Understanding relationships between concepts\n"
    "   - 3 points: Applying knowledge to identify correct terms in context\n"
    "   - 4-5 points: Complex concepts requiring deeper understanding\n"
    "   - More foundational/important concepts should have HIGHER scores.\n\n"
    "4. QUIZ PARAMETERS:\n"
    "   - max_attempts: 2-3 for easy quizzes (3-4 questions), 3-4 for harder quizzes (5+ questions)\n"
    "   - max_execution_time: Calculate as (number_of_questions × 45 seconds). Example: 4 questions = 180 seconds.\n"
    "   - min_pass_score: Set to 50-60%% of total score (allows some mistakes but ensures basic understanding).\n\n"
    "Output strictly as minified JSON with keys: questions (3-5), min_pass_score, max_attempts, max_execution_time.\n"
    "CRITICAL: You have a maximum of 10000 tokens. Ensure your JSON response is COMPLETE and VALID within this limit.\n"
    "If approaching the limit, generate fewer questions but ensure the JSON is properly closed with all required fields.\n"
    "Schema:\n"
    "{{\n"
    "  \"questions\": [ {{ \"question\": string, \"answers\": [string, ...], \"score\": integer }}, ... ],\n"
    "  \"min_pass_score\": number,\n"
    "  \"max_attempts\": integer,\n"
    "  \"max_execution_time\": integer\n"
    "}}\n"
    "Rules: questions must be SHORT ANSWERS in {language_name}; answers are acceptable synonyms in {language_name}; "
    "scores range 1-5 based on importance; min_pass_score should be 50-60%% of sum(scores); max_attempts 2-4."
)

PROMPT_GIFTQUIZ_REASONING_QUESTIONS_TEMPLATE: str = (
    "You are an assessment designer. Generate REASONING questions in {language_name} (language code: {course_language}) in strict JSON.\n"
    "REASONING questions present real-world scenarios or dilemmas that require thoughtful analysis and multi-step reasoning.\n"
    "Each question should describe a situation or challenge, and the answer should explain the reasoning process and recommended actions.\n"
    "Prioritize the CURRENT LESSON materials, use previous lesson summaries only if helpful.\n"
    "IMPORTANT: All questions and answers must be written in {language_name}.\n\n"
    "{difficulty_block}"
    "Materials (may be partial):\n"
    "Gift (raw, may include questions):\n{gift_text}\n\n"
    "{extracted_block}"
    "{other_block}"
    "{prev_block}"
    "=== REASONING QUESTION GUIDELINES ===\n\n"
    "1. SCENARIO DIVERSITY: Each question MUST present a UNIQUE scenario testing different aspects.\n"
    "   - Cover various real-world applications of the lesson concepts.\n"
    "   - Scenarios should be practical and relatable to learners.\n"
    "   - Avoid asking the same type of reasoning multiple times.\n\n"
    "2. ANSWER QUALITY: The answer should be a comprehensive reasoning explanation.\n"
    "   - Include step-by-step thought process.\n"
    "   - Explain WHY the recommended actions are appropriate.\n"
    "   - Reference relevant concepts from the lesson.\n\n"
    "3. SCORE WEIGHTING (2-5 points per question - reasoning is harder):\n"
    "   - 2 points: Simple scenario with straightforward reasoning\n"
    "   - 3 points: Moderate complexity requiring multiple considerations\n"
    "   - 4 points: Complex scenario with trade-offs to analyze\n"
    "   - 5 points: Advanced scenario requiring synthesis of multiple concepts\n\n"
    "4. QUIZ PARAMETERS:\n"
    "   - max_attempts: 3-4 (reasoning questions are harder, allow more attempts)\n"
    "   - max_execution_time: Calculate as (number_of_questions × 90 seconds). Reasoning takes longer.\n"
    "   - min_pass_score: Set to 50-60%% of total score.\n\n"
    "Output strictly as minified JSON with keys: questions (2-4), min_pass_score, max_attempts, max_execution_time.\n"
    "CRITICAL: You have a maximum of 10000 tokens. Ensure your JSON response is COMPLETE and VALID within this limit.\n"
    "If approaching the limit, generate fewer questions but ensure the JSON is properly closed with all required fields.\n"
    "Schema:\n"
    "{{\n"
    "  \"questions\": [ {{ \"question\": string, \"answers\": [string], \"score\": integer }}, ... ],\n"
    "  \"min_pass_score\": number,\n"
    "  \"max_attempts\": integer,\n"
    "  \"max_execution_time\": integer\n"
    "}}\n"
    "Rules: questions must be REASONING QUESTIONS in {language_name} that present scenarios requiring analysis;\n"
    "each question should have ONE answer in the answers array containing the reasoning explanation and recommended actions;\n"
    "scores range 2-5 based on complexity; min_pass_score should be 50-60%% of sum(scores); max_attempts 3-4.\n"
    "Example question format: \"During [scenario], [situation occurs]. How do you react?\"\n"
    "Example answer format: \"[Step-by-step reasoning and recommended actions based on the scenario].\""
)

PROMPT_GIFTQUIZ_EXTRACTED_BLOCK_TEMPLATE: str = "Extracted file text:\n{extracted}\n\n"
PROMPT_GIFTQUIZ_OTHER_BLOCK_TEMPLATE: str = "Other lesson topics (title: excerpt):\n{other_texts}\n\n"
PROMPT_GIFTQUIZ_PREV_BLOCK_TEMPLATE: str = "Previous lessons summaries:\n{prev_text}\n\n"

# ---------------------------------------------------------------------------
# ANSWER VALIDATION PROMPTS: imported from quiz_answer_validation.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def _init_logs():
    if "api_logs" not in st.session_state:
        st.session_state.api_logs = []


def _log_api_call(
    phase: str,
    system_prompt: str,
    user_prompt: str,
    raw_response: str,
    parsed_response: object,
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: float,
    metadata: Optional[dict] = None,
):
    _init_logs()
    entry = {
        "timestamp": _now_iso(),
        "phase": phase,
        "request": {
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        "response": {
            "raw": raw_response,
            "parsed": parsed_response,
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "duration_ms": round(duration_ms, 1),
    }
    if metadata:
        entry["metadata"] = metadata
    st.session_state.api_logs.append(entry)


def _format_logs_prompt_stream(entries: list) -> str:
    """Human-readable chronological dump of every logged API call with full prompts."""
    lines: list[str] = []
    for i, e in enumerate(entries, 1):
        lines.append("=" * 72)
        lines.append(
            f"[{i}] {e.get('timestamp', '')}  phase={e.get('phase', '')}"
        )
        req = e.get("request") or {}
        lines.append(f"model: {req.get('model', '')}")
        if e.get("metadata"):
            lines.append(
                "metadata: "
                + json.dumps(e["metadata"], ensure_ascii=False)
            )
        usage = e.get("usage") or {}
        lines.append(
            f"tokens: in={usage.get('input_tokens', '')} "
            f"out={usage.get('output_tokens', '')} "
            f"duration_ms={e.get('duration_ms', '')}"
        )
        lines.append("")
        lines.append("--- SYSTEM PROMPT ---")
        lines.append(str(req.get("system_prompt") or ""))
        lines.append("")
        lines.append("--- USER PROMPT ---")
        lines.append(str(req.get("user_prompt") or ""))
        lines.append("")
        lines.append("--- RAW RESPONSE ---")
        resp = e.get("response") or {}
        lines.append(str(resp.get("raw") or ""))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OPENROUTER CALL
# ---------------------------------------------------------------------------

def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str,
    max_tokens: int = OPENROUTER_MAX_TOKENS,
    temperature: float = 0.3,
    top_p: float = 0.8,
    retries: int = 3,
    extra_body: Optional[dict] = None,
) -> tuple[str, str, int, int, float]:
    """Returns (content, model_used, input_tokens, output_tokens, duration_ms)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            duration_ms = (time.perf_counter() - t0) * 1000
            content = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            in_t = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_t = int(getattr(usage, "completion_tokens", 0) or 0)
            resp_model = str(getattr(resp, "model", None) or model)
            return content, resp_model, in_t, out_t, duration_ms
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise last_err if last_err else RuntimeError("OpenRouter call failed")


# ---------------------------------------------------------------------------
# HELPERS (from prompt_review_runner_2.py)
# ---------------------------------------------------------------------------

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


def _unwrap_outer_json_braces(s: str) -> str:
    """Remove redundant outer {{ ... }} when models mirror doubled braces from prompts."""
    s = (s or "").strip()
    while len(s) >= 4 and s.startswith("{{") and s.endswith("}}"):
        s = s[1:-1].strip()
    return s


def safe_json_loads(raw: str) -> Optional[dict]:
    if not isinstance(raw, str):
        return None

    def _try(s: str):
        if not isinstance(s, str) or not s.strip():
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    candidates: list[str] = []
    for base in (raw.strip(), _strip_code_fences(raw)):
        if base and base not in candidates:
            candidates.append(base)
        u = _unwrap_outer_json_braces(base)
        if u and u not in candidates:
            candidates.append(u)

    for s in candidates:
        obj = _try(s)
        if obj is not None:
            return obj

    try:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            for s in (match.group(0), _unwrap_outer_json_braces(match.group(0))):
                obj = _try(s)
                if obj is not None:
                    return obj
    except Exception:
        pass
    return None


def normalize_short_answers(obj: Optional[dict]) -> tuple[list[dict], dict]:
    questions: list[dict] = []
    quiz: dict = {}
    if isinstance(obj, dict):
        raw_q = obj.get("questions") or []
        if isinstance(raw_q, list):
            for it in raw_q:
                try:
                    q = str((it or {}).get("question") or "").strip()
                    ans = (it or {}).get("answers") or []
                    sc = (it or {}).get("score")
                    if not q or not isinstance(ans, list):
                        continue
                    clean_ans = []
                    for a in ans:
                        s = str(a or "").strip()
                        if s and s not in clean_ans:
                            clean_ans.append(s)
                    if not clean_ans:
                        continue
                    score = int(sc) if isinstance(sc, (int, float)) else 1
                    if score <= 0:
                        score = 1
                    questions.append(
                        {"question": q, "answers": clean_ans, "score": score})
                except Exception:
                    continue
        quiz = {
            "min_pass_score": obj.get("min_pass_score"),
            "max_attempts": obj.get("max_attempts"),
            "max_execution_time": obj.get("max_execution_time"),
        }

    total_score = sum(int(q.get("score", 1)) for q in questions)
    num_questions = len(questions)

    mps = quiz.get("min_pass_score")
    try:
        mps_val = float(mps) if mps is not None else None
    except Exception:
        mps_val = None
    if mps_val is None or mps_val >= total_score or mps_val < 0:
        mps_val = max(1.0, round(total_score * 0.55, 2)
                      ) if total_score > 0 else 1.0
        if mps_val >= total_score and total_score > 0:
            mps_val = max(1.0, total_score - 1.0)

    att = quiz.get("max_attempts")
    att_val = int(att) if isinstance(att, (int, float)) and 2 <= int(
        att) <= 5 else (3 if num_questions <= 4 else 4)

    met = quiz.get("max_execution_time")
    if isinstance(met, (int, float)) and int(met) >= 30:
        met_val = int(met)
        met_val = max(num_questions * 30, min(met_val, num_questions * 120))
    else:
        met_val = max(60, num_questions * 45)

    return questions, {"min_pass_score": float(mps_val), "max_attempts": int(att_val), "max_execution_time": int(met_val)}


# ---------------------------------------------------------------------------
# PROMPT BUILDERS
# ---------------------------------------------------------------------------

def build_system_prompt(course_language: str) -> str:
    return PROMPT_SYSTEM_TEMPLATE.format(
        language_name=get_language_name(course_language),
        course_language=course_language,
    )


def build_giftquiz_questions_prompt(payload: dict, course_language: str = "en", difficulty_level: int = 3) -> str:
    language_name = get_language_name(course_language)
    difficulty_cfg = get_difficulty_config(difficulty_level)
    data = payload.get("data") or {}
    gift_text = (data.get("gift") or "")[:4000]
    extracted = (data.get("extracted_text") or "")[:4000]
    other = data.get("lesson_other_topics") or []
    other_texts = []
    for item in other[:50]:
        title = str(item.get("title") or "")
        text = str(item.get("text") or "")
        if text:
            other_texts.append(f"- {title}: {text[:1000]}")
    prev_summaries = data.get("previous_lesson_summaries") or []
    prev_text = "\n".join(str(s)[:1000] for s in prev_summaries[:20])

    extracted_block = PROMPT_GIFTQUIZ_EXTRACTED_BLOCK_TEMPLATE.format(
        extracted=extracted) if extracted else ""
    other_block = PROMPT_GIFTQUIZ_OTHER_BLOCK_TEMPLATE.format(
        other_texts="\n".join(other_texts)) if other_texts else ""
    prev_block = PROMPT_GIFTQUIZ_PREV_BLOCK_TEMPLATE.format(
        prev_text=prev_text) if prev_text else ""

    return PROMPT_GIFTQUIZ_QUESTIONS_TEMPLATE.format(
        language_name=language_name, course_language=course_language,
        gift_text=gift_text, extracted_block=extracted_block,
        other_block=other_block, prev_block=prev_block,
        difficulty_block=difficulty_cfg["question_generation_instruction"],
    )


def build_giftquiz_reasoning_questions_prompt(payload: dict, course_language: str = "en", difficulty_level: int = 3) -> str:
    language_name = get_language_name(course_language)
    difficulty_cfg = get_difficulty_config(difficulty_level)
    data = payload.get("data") or {}
    gift_text = (data.get("gift") or "")[:4000]
    extracted = (data.get("extracted_text") or "")[:4000]
    other = data.get("lesson_other_topics") or []
    other_texts = []
    for item in other[:50]:
        title = str(item.get("title") or "")
        text = str(item.get("text") or "")
        if text:
            other_texts.append(f"- {title}: {text[:1000]}")
    prev_summaries = data.get("previous_lesson_summaries") or []
    prev_text = "\n".join(str(s)[:1000] for s in prev_summaries[:20])

    extracted_block = PROMPT_GIFTQUIZ_EXTRACTED_BLOCK_TEMPLATE.format(
        extracted=extracted) if extracted else ""
    other_block = PROMPT_GIFTQUIZ_OTHER_BLOCK_TEMPLATE.format(
        other_texts="\n".join(other_texts)) if other_texts else ""
    prev_block = PROMPT_GIFTQUIZ_PREV_BLOCK_TEMPLATE.format(
        prev_text=prev_text) if prev_text else ""

    return PROMPT_GIFTQUIZ_REASONING_QUESTIONS_TEMPLATE.format(
        language_name=language_name, course_language=course_language,
        gift_text=gift_text, extracted_block=extracted_block,
        other_block=other_block, prev_block=prev_block,
        difficulty_block=difficulty_cfg["reasoning_generation_instruction"],
    )


def _quiz_summary_from_payload(payload: dict) -> str:
    data = payload.get("data") or {}
    title = (data.get("title") or "Quiz").strip()
    intro = (data.get("introduction") or "").strip()
    gift = (data.get("gift") or "").strip()
    parts = [p for p in (title, intro) if p]
    if gift and gift not in " ".join(parts):
        parts.append(gift[:280])
    return ". ".join(parts) if parts else "Quiz"


def _rephrase_conversation_upto_question(answers: dict, q_idx: int) -> list[dict]:
    hist: list[dict] = []
    for j in range(q_idx + 1):
        for m in answers.get(j, {}).get("conversation", []):
            hist.append(dict(m))
    return hist


def _conversation_history_prior_questions(answers: dict, q_idx: int) -> list[dict]:
    hist: list[dict] = []
    for j in range(q_idx):
        for m in answers.get(j, {}).get("conversation", []):
            hist.append(dict(m))
    return hist


def _build_completion_summary(
    questions: list,
    quiz_cfg: dict,
    answers: dict,
    total_earned: float,
    total_possible: float,
    passed: bool,
) -> str:
    base = st.session_state.get("quiz_summary") or "Quiz"
    lines = [
        f"{base}",
        f"Questions: {len(questions)}",
        f"Score: {total_earned:.1f} / {total_possible}",
        f"Min pass score: {quiz_cfg.get('min_pass_score', 0)}",
        f"Passed: {passed}",
    ]
    if total_possible and total_possible > 0:
        r = max(0.0, min(1.0, float(total_earned) / float(total_possible)))
        lines.append(f"Performance ratio (earned/max): {r:.4f}")
    else:
        lines.append("Performance ratio (earned/max): unavailable (no maximum points)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# DEFAULT PAYLOADS (from prompt_review_runner_2.py — only GiftQuiz topics)
# ---------------------------------------------------------------------------
DEFAULT_PAYLOADS: list[dict] = [
    {
        "label": "Quiz #1283 — Quantum Computer (reasoning)",
        "type": "topic",
        "course_id": 27,
        "entity_id": 1283,
        "topic_type": "EscolaLms\\TopicTypeGift\\Models\\GiftQuiz",
        "course_language": "en",
        "data": {
            "title": "Quiz",
            "introduction": "Ok12",
            "topicable_class": "EscolaLms\\TopicTypeGift\\Models\\GiftQuiz",
            "gift": "theProject",
            "topic_json": {"reasoningQuiz": True},
            "questions_count": 4,
            "questions": [
                {"question": "What type of computer performs calculations using the principles of quantum mechanics? {=Quantum computer=A quantum computer}",
                    "type": "short_answers", "max_score": 25},
                {"question": "What is the basic unit of information in a quantum computer, which can represent more than just 0 or 1? {=Qubit=Qubits}",
                    "type": "short_answers", "max_score": 25},
                {"question": "In contrast to quantum computers, what is the basic unit of information in classical computers? {=Bit=Bits}",
                    "type": "short_answers", "max_score": 25},
                {"question": "What field of physics provides the principles that quantum computers use for calculations? {=Quantum mechanics=The principles of quantum mechanics}",
                    "type": "short_answers", "max_score": 25},
            ],
            "lesson_other_topics": [
                {
                    "title": "A quantum computer",
                    "topicable_type": "EscolaLms\\TopicTypes\\Models\\TopicContent\\RichText",
                    "text": (
                        "A quantum computer is a type of computer that uses the principles of "
                        "quantum mechanics to perform calculations. Unlike classical computers "
                        "that use bits to represent either a 0 or a 1, quantum computers use "
                        "qubits which can represent 0, 1, or both simultaneously through a property "
                        "called superposition."
                    ),
                }
            ],
            "previous_lesson_summaries": [],
        },
    },
    {
        "label": "Quiz #1369 — Quantum State (short answers)",
        "type": "topic",
        "course_id": 27,
        "entity_id": 1369,
        "topic_type": "EscolaLms\\TopicTypeGift\\Models\\GiftQuiz",
        "course_language": "en",
        "data": {
            "title": "Quiz",
            "topicable_class": "EscolaLms\\TopicTypeGift\\Models\\GiftQuiz",
            "gift": "theProject",
            "questions_count": 5,
            "questions": [
                {"question": "What term describes the mathematical representation of a quantum system that holds all possible information about it? {=Quantum state}",
                    "type": "short_answers", "max_score": 10},
                {"question": "What is the basic unit of quantum information, analogous to a bit in classical computing? {=Qubit=Quantum bit}",
                    "type": "short_answers", "max_score": 10},
                {"question": "What quantum principle allows a qubit to be in a combination of both 0 and 1 states at the same time? {=Superposition}",
                    "type": "short_answers", "max_score": 10},
                {"question": "What is the name for the quantum phenomenon where multiple qubits are linked and their states are correlated, no matter how far apart they are? {=Entanglement=Quantum entanglement}",
                    "type": "short_answers", "max_score": 10},
                {"question": "According to the lesson material, what is one mathematical object used to represent a quantum state? {=State vector=Wavefunction}",
                    "type": "short_answers", "max_score": 10},
            ],
            "lesson_other_topics": [
                {
                    "title": "A quantum state",
                    "topicable_type": "EscolaLms\\TopicTypes\\Models\\TopicContent\\RichText",
                    "text": (
                        "A quantum state is a mathematical description of a quantum system, like "
                        "a particle, that contains all possible information about it. It is "
                        "represented by a state vector or wavefunction and determines the "
                        "probabilities of the system's properties when measured."
                    ),
                }
            ],
            "previous_lesson_summaries": [
                "This lesson introduces quantum computing, which uses quantum-mechanical phenomena for calculations."
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------------------

def _init_state():
    _init_logs()
    defaults = {
        "stage": "setup",          # setup | generating | quiz | results
        "questions": [],           # generated questions
        "quiz_cfg": {},            # min_pass_score, max_attempts, max_execution_time
        "current_q_idx": 0,
        # {q_idx: {"user_answer": str, "result": dict, "attempts": int, "conversation": []}}
        "answers": {},
        "total_score": 0.0,
        "payload_json_text": "",
        "is_reasoning": False,
        "difficulty_level": 3,     # 1-5, default intermediate
        "course_language": "en",   # ISO 639-1 code
        "quiz_summary": "",
        "rephrase_cache": {},      # key -> conversational question text
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, fall back to os.getenv."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


def main():
    st.set_page_config(page_title="Quiz Runner", page_icon="📝", layout="wide")
    _init_state()

    api_key = _get_secret("OPENROUTER_API_KEY")
    model = _get_secret("OPENROUTER_MODEL", "google/gemini-2.5-pro")
    validation_model_default = _get_secret(
        "OPENROUTER_VALIDATION_MODEL", "openai/gpt-4o")
    rephrase_model_default = _get_secret(
        "OPENROUTER_REPHRASE_MODEL", "google/gemini-3.1-flash-lite-preview")
    followup_model_default = _get_secret(
        "OPENROUTER_FOLLOWUP_MODEL", "google/gemini-3.1-flash-lite-preview")
    completion_model_default = _get_secret(
        "OPENROUTER_COMPLETION_MODEL", "google/gemini-3.1-flash-lite-preview")

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        st.subheader("Models (OpenRouter IDs)")
        model = st.text_input(
            "1. Question generation",
            value=model,
            help="Default: OPENROUTER_MODEL or google/gemini-2.5-pro",
        )
        validation_model = st.text_input(
            "2. Answer validation (JSON)",
            value=validation_model_default,
            help="Default: OPENROUTER_VALIDATION_MODEL or openai/gpt-4o",
        )
        rephrase_model = st.text_input(
            "3. Question rephrasing (spoken intro)",
            value=rephrase_model_default,
            help="Default: OPENROUTER_REPHRASE_MODEL or google/gemini-3.1-flash-lite-preview",
        )
        followup_model = st.text_input(
            "4. Follow-up tutor (after failed attempt)",
            value=followup_model_default,
            help="Default: OPENROUTER_FOLLOWUP_MODEL or google/gemini-3.1-flash-lite-preview",
        )
        completion_model = st.text_input(
            "5. Completion message (end of quiz)",
            value=completion_model_default,
            help="Default: OPENROUTER_COMPLETION_MODEL or google/gemini-3.1-flash-lite-preview",
        )

        st.divider()

        # Logs section
        st.subheader("API Logs")
        st.caption(f"{len(st.session_state.api_logs)} call(s) recorded")

        if st.session_state.api_logs:
            logs_json = json.dumps(
                st.session_state.api_logs, ensure_ascii=False, indent=2)
            logs_txt = _format_logs_prompt_stream(st.session_state.api_logs)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Download full logs (JSON)",
                data=logs_json,
                file_name=f"quiz_logs_{ts}.json",
                mime="application/json",
            )
            st.download_button(
                "Download prompt stream (TXT)",
                data=logs_txt,
                file_name=f"quiz_prompt_stream_{ts}.txt",
                mime="text/plain",
                help="Chronological text: every phase with full system/user prompts and raw responses.",
            )
            if st.button("Clear logs"):
                st.session_state.api_logs = []
                st.rerun()

        st.divider()
        if st.button("Reset quiz", type="secondary"):
            for k in list(st.session_state.keys()):
                if k not in ("api_logs",):
                    del st.session_state[k]
            _init_state()
            st.rerun()

    # ── Main area ─────────────────────────────────────────────────────────
    st.title("Quiz Runner")

    if st.session_state.stage == "setup":
        _render_setup()
    elif st.session_state.stage == "generating":
        _render_generating(api_key, model)
    elif st.session_state.stage == "quiz":
        _render_quiz(
            api_key,
            validation_model,
            rephrase_model,
            followup_model,
        )
    elif st.session_state.stage == "results":
        _render_results(api_key, completion_model)


# ---------------------------------------------------------------------------
# SETUP STAGE
# ---------------------------------------------------------------------------

def _render_setup():
    st.subheader("1. Course language & difficulty")

    col_lang, col_diff = st.columns(2)

    with col_lang:
        lang_options = {v: k for k, v in LANGUAGE_MAP.items()}
        current_lang_code = st.session_state.get("course_language", "en")
        current_lang_name = get_language_name(current_lang_code)
        selected_lang_name = st.selectbox(
            "Course language",
            options=list(lang_options.keys()),
            index=list(lang_options.keys()).index(
                current_lang_name) if current_lang_name in lang_options else 0,
            help="Language for question generation and validation feedback",
        )
        st.session_state.course_language = lang_options[selected_lang_name]

    with col_diff:
        difficulty_options = {v["label"]: k for k,
                              v in DIFFICULTY_LEVELS.items()}
        current_level = st.session_state.get("difficulty_level", 3)
        current_label = DIFFICULTY_LEVELS[current_level]["label"]

        selected_label = st.select_slider(
            "Difficulty level",
            options=list(difficulty_options.keys()),
            value=current_label,
            help=(
                "Level 1 (Beginner): very easy questions, maximum tolerance for imprecise answers. "
                "Level 5 (Expert): very hard questions, answers must be precise."
            ),
        )
        st.session_state.difficulty_level = difficulty_options[selected_label]

    diff_cfg = get_difficulty_config(st.session_state.difficulty_level)
    threshold_pct = int(diff_cfg["validation_threshold"] * 100)
    st.caption(
        f"Passing threshold for validation: **{threshold_pct}%** match required")

    st.divider()
    st.subheader("2. Select or paste a payload")

    tab_preset, tab_custom = st.tabs(["Preset payloads", "Custom JSON"])

    with tab_preset:
        labels = [p["label"] for p in DEFAULT_PAYLOADS]
        choice = st.selectbox("Choose a preset quiz payload", labels)
        idx = labels.index(choice)
        selected = DEFAULT_PAYLOADS[idx]
        st.json(selected, expanded=False)
        if st.button("Use this payload", key="btn_preset"):
            st.session_state.payload_json_text = json.dumps(
                selected, ensure_ascii=False, indent=2)
            st.session_state.stage = "generating"
            st.rerun()

    with tab_custom:
        custom = st.text_area("Paste payload JSON", height=300,
                              placeholder='{"type": "topic", "course_language": "en", "data": {...}}')
        if st.button("Use custom payload", key="btn_custom"):
            try:
                parsed = json.loads(custom)
                st.session_state.payload_json_text = json.dumps(
                    parsed, ensure_ascii=False, indent=2)
                st.session_state.stage = "generating"
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")


# ---------------------------------------------------------------------------
# GENERATING STAGE
# ---------------------------------------------------------------------------

def _render_generating(api_key: str, model: str):
    payload = json.loads(st.session_state.payload_json_text)
    # User's explicit language selection in setup takes priority over payload
    course_language = st.session_state.get("course_language", "en")

    data = payload.get("data") or {}
    topic_json = data.get("topic_json") or {}
    is_reasoning = bool(topic_json.get("reasoningQuiz"))
    st.session_state.is_reasoning = is_reasoning

    difficulty_level = st.session_state.get("difficulty_level", 3)
    diff_cfg = get_difficulty_config(difficulty_level)

    quiz_type = "REASONING" if is_reasoning else "SHORT ANSWER"
    st.info(
        f"Generating **{quiz_type}** questions for entity_id={payload.get('entity_id')} "
        f"| Difficulty: **{diff_cfg['label']}** …"
    )

    if not api_key:
        st.error("Please provide an OpenRouter API key in the sidebar.")
        if st.button("Back"):
            st.session_state.stage = "setup"
            st.rerun()
        return

    system_prompt = build_system_prompt(course_language)
    if is_reasoning:
        user_prompt = build_giftquiz_reasoning_questions_prompt(
            payload, course_language, difficulty_level)
    else:
        user_prompt = build_giftquiz_questions_prompt(
            payload, course_language, difficulty_level)

    with st.spinner("Calling LLM to generate questions..."):
        try:
            raw, model_used, in_t, out_t, dur = call_openrouter(
                system_prompt, user_prompt, api_key, model,
            )
        except Exception as e:
            st.error(f"API call failed: {e}")
            if st.button("Back"):
                st.session_state.stage = "setup"
                st.rerun()
            return

    _log_api_call(
        phase="generate_questions",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=raw,
        parsed_response=None,
        model=model_used,
        input_tokens=in_t,
        output_tokens=out_t,
        duration_ms=dur,
    )

    parsed = safe_json_loads(raw)
    if not parsed:
        st.error("Failed to parse LLM response as JSON.")
        st.code(raw, language="json")
        if st.button("Back"):
            st.session_state.stage = "setup"
            st.rerun()
        return

    questions, quiz_cfg = normalize_short_answers(parsed)
    if not questions:
        st.error("No valid questions generated.")
        st.code(raw, language="json")
        if st.button("Back"):
            st.session_state.stage = "setup"
            st.rerun()
        return

    # Update the log with parsed data
    st.session_state.api_logs[-1]["response"]["parsed"] = {
        "questions": questions,
        "quiz_cfg": quiz_cfg,
    }

    st.session_state.questions = questions
    st.session_state.quiz_cfg = quiz_cfg
    st.session_state.current_q_idx = 0
    st.session_state.answers = {}
    st.session_state.quiz_summary = _quiz_summary_from_payload(payload)
    st.session_state.rephrase_cache = {}
    st.session_state.pop("completion_message_for_session", None)
    st.session_state.stage = "quiz"
    st.rerun()


# ---------------------------------------------------------------------------
# QUIZ STAGE
# ---------------------------------------------------------------------------

def _render_quiz(
    api_key: str,
    validation_model: str,
    rephrase_model: str,
    followup_model: str,
):
    questions = st.session_state.questions
    quiz_cfg = st.session_state.quiz_cfg
    q_idx = st.session_state.current_q_idx

    if q_idx >= len(questions):
        st.session_state.stage = "results"
        st.rerun()
        return

    q = questions[q_idx]
    total_q = len(questions)
    max_attempts = quiz_cfg.get("max_attempts", 3)
    course_language = st.session_state.get("course_language", "en")

    difficulty_level = st.session_state.get("difficulty_level", 3)
    diff_cfg = get_difficulty_config(difficulty_level)
    pass_threshold = diff_cfg["validation_threshold"]

    if q_idx not in st.session_state.answers:
        st.session_state.answers[q_idx] = {
            "attempts": 0,
            "best_score": 0.0,
            "best_validation_score": 0.0,
            "conversation": [],
            "passed": False,
            "skipped": False,
        }

    ans_state = st.session_state.answers[q_idx]

    # Progress bar
    progress = q_idx / total_q
    st.progress(progress, text=f"Question {q_idx + 1} of {total_q}")

    # Quiz config info
    cols = st.columns(5)
    cols[0].metric("Min pass score", quiz_cfg.get("min_pass_score", "?"))
    cols[1].metric("Max attempts", max_attempts)
    cols[2].metric("Question score", q["score"])
    cols[3].metric("Attempts used", ans_state["attempts"])
    cols[4].metric("Difficulty", diff_cfg["label"])
    st.caption(
        "Clarification and off-topic messages do not count toward max attempts."
    )

    st.divider()

    quiz_type_label = "Reasoning" if st.session_state.is_reasoning else "Short Answer"
    st.subheader(f"[{quiz_type_label}] Question {q_idx + 1}")

    quiz_summary = st.session_state.get("quiz_summary") or "Quiz"
    rcache = st.session_state.rephrase_cache
    r_key = f"{q_idx}:{len(ans_state['conversation'])}:{ans_state['attempts']}"
    display_q = q["question"]
    if r_key in rcache:
        display_q = rcache[r_key]
    elif api_key:
        is_first = q_idx == 0
        not_passed = bool(ans_state["conversation"]) and not ans_state["passed"]
        conv_hist = _rephrase_conversation_upto_question(
            st.session_state.answers, q_idx)
        usr_rep = build_rephraser_user_prompt(
            original_question=q["question"],
            quiz_summary=quiz_summary,
            is_first_question=is_first,
            conversation_history=conv_hist,
            language=course_language,
            not_passed=not_passed,
        )
        try:
            with st.spinner("Preparing question wording..."):
                raw_rep, m_rep, in_r, out_r, dur_r = call_openrouter(
                    QUIZ_REPHRASER_SYSTEM_PROMPT,
                    usr_rep,
                    api_key,
                    rephrase_model,
                    max_tokens=2000,
                    temperature=0.85,
                    top_p=0.9,
                )
            _log_api_call(
                phase=f"rephrase_q{q_idx}",
                system_prompt=QUIZ_REPHRASER_SYSTEM_PROMPT,
                user_prompt=usr_rep,
                raw_response=raw_rep,
                parsed_response=None,
                model=m_rep,
                input_tokens=in_r,
                output_tokens=out_r,
                duration_ms=dur_r,
            )
            if raw_rep.strip():
                rcache[r_key] = raw_rep.strip()
                display_q = rcache[r_key]
        except Exception as e:
            st.caption(f"Question rephrase skipped: {e}")

    st.caption("Original (reference)")
    with st.expander("Show raw quiz question text", expanded=False):
        st.markdown(q["question"])
    st.markdown(f"**{display_q}**")

    # Conversation history
    if ans_state["conversation"]:
        st.caption("Conversation:")
        for msg in ans_state["conversation"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

    # Check limits
    if ans_state["passed"]:
        st.success(
            f"Correct! Score: {ans_state['best_score']:.1f}/{q['score']}")
        if st.button("Next question", key=f"next_{q_idx}"):
            st.session_state.current_q_idx = q_idx + 1
            st.rerun()
        return

    if ans_state["attempts"] >= max_attempts:
        st.warning(
            f"Max attempts reached. Best score: {ans_state['best_score']:.1f}/{q['score']}")
        if st.button("Next question", key=f"next_{q_idx}"):
            st.session_state.current_q_idx = q_idx + 1
            st.rerun()
        return

    # Answer input (key uses conversation length so it refreshes after non-counting turns)
    _input_key = f"answer_input_{q_idx}_{len(ans_state['conversation'])}"
    user_answer = st.text_area(
        "Your answer:", key=_input_key, height=100)

    col_submit, col_skip = st.columns([1, 1])

    with col_skip:
        if st.button("Skip question", key=f"skip_{q_idx}"):
            ans_state["skipped"] = True
            st.session_state.current_q_idx = q_idx + 1
            st.rerun()

    with col_submit:
        if st.button("Submit answer", key=f"submit_{q_idx}", type="primary"):
            if not user_answer.strip():
                st.warning("Please enter an answer.")
                return
            if not api_key:
                st.error("Please provide an OpenRouter API key in the sidebar.")
                return

            ans_state["conversation"].append(
                {"role": "user", "content": user_answer.strip()})
            turn_idx = len(ans_state["conversation"])

            val_system = build_answer_validation_system_prompt(
                diff_cfg["validation_instruction"])
            val_user = build_answer_validation_user_prompt(
                quiz_question=q["question"],
                user_answer=user_answer.strip(),
                correct_answers=q["answers"],
                conversation_history=ans_state["conversation"],
                max_possible_score=float(q["score"]),
                language=course_language,
            )

            with st.spinner("Validating your answer..."):
                try:
                    raw_val, m_used, in_t, out_t, dur = call_openrouter(
                        val_system,
                        val_user,
                        api_key,
                        validation_model,
                        max_tokens=2000,
                        extra_body={"reasoning": {"effort": "medium"}},
                    )
                except Exception as e:
                    st.error(f"Validation API call failed: {e}")
                    ans_state["conversation"].pop()
                    return

            val_parsed = safe_json_loads(raw_val)
            if not val_parsed:
                st.error("Failed to parse validation response.")
                st.code(raw_val)
                ans_state["conversation"].pop()
                return

            v_score = float(val_parsed.get("validation_score", 0.0))
            a_score = float(val_parsed.get("answer_score", 0.0))
            user_intent = val_parsed.get("user_intent", "answer_attempt")
            intent_norm = _normalize_user_intent_for_quota(user_intent)
            exempt_from_attempt = intent_norm in _INTENTS_EXEMPT_FROM_ATTEMPT_QUOTA
            if not exempt_from_attempt:
                ans_state["attempts"] += 1

            _log_api_call(
                phase=f"validate_q{q_idx}_turn{turn_idx}",
                system_prompt=val_system,
                user_prompt=val_user,
                raw_response=raw_val,
                parsed_response=val_parsed,
                model=m_used,
                input_tokens=in_t,
                output_tokens=out_t,
                duration_ms=dur,
                metadata={
                    "question_index": q_idx,
                    "user_intent": str(user_intent),
                    "counts_as_quiz_attempt": not exempt_from_attempt,
                },
            )

            validation_error = val_parsed.get("validation_error", "")

            if a_score > ans_state["best_score"]:
                ans_state["best_score"] = a_score
                ans_state["best_validation_score"] = v_score

            if v_score >= pass_threshold:
                ans_state["passed"] = True
                st.rerun()
                return

            followup_text = ""
            curr_msgs = list(ans_state["conversation"])
            prior_all = _conversation_history_prior_questions(
                st.session_state.answers, q_idx)
            all_hist = prior_all + curr_msgs
            fu_user = build_followup_user_prompt(
                quiz_question=q["question"],
                user_answer=user_answer.strip(),
                validation_error=validation_error or "",
                validation_score=v_score,
                conversation_history_all_questions=all_hist,
                current_question_messages=curr_msgs,
                language=course_language,
                user_intent=str(user_intent),
            )
            with st.spinner("Generating tutor follow-up..."):
                try:
                    raw_fu, m_fu, in_fu, out_fu, dur_fu = call_openrouter(
                        QUIZ_FOLLOWUP_SYSTEM_PROMPT,
                        fu_user,
                        api_key,
                        followup_model,
                        max_tokens=2000,
                        temperature=0.85,
                        top_p=0.9,
                    )
                    _log_api_call(
                        phase=f"followup_q{q_idx}_turn{turn_idx}",
                        system_prompt=QUIZ_FOLLOWUP_SYSTEM_PROMPT,
                        user_prompt=fu_user,
                        raw_response=raw_fu,
                        parsed_response=None,
                        model=m_fu,
                        input_tokens=in_fu,
                        output_tokens=out_fu,
                        duration_ms=dur_fu,
                        metadata={
                            "question_index": q_idx,
                            "user_intent": str(user_intent),
                        },
                    )
                    followup_text = (raw_fu or "").strip()
                except Exception as e:
                    st.caption(f"Tutor follow-up skipped: {e}")

            feedback = followup_text or validation_error or "Try again!"
            ans_state["conversation"].append(
                {"role": "assistant", "content": feedback})
            st.rerun()


# ---------------------------------------------------------------------------
# RESULTS STAGE
# ---------------------------------------------------------------------------

def _render_results(api_key: str, completion_model: str):
    questions = st.session_state.questions
    quiz_cfg = st.session_state.quiz_cfg
    answers = st.session_state.answers

    st.subheader("Quiz Results")

    total_possible = sum(q["score"] for q in questions)
    total_earned = sum(answers.get(i, {}).get("best_score", 0.0)
                       for i in range(len(questions)))
    min_pass = quiz_cfg.get("min_pass_score", 0)
    passed = total_earned >= min_pass

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Score", f"{total_earned:.1f} / {total_possible}")
    col2.metric("Min Pass Score", f"{min_pass}")
    col3.metric("Result", "PASSED" if passed else "NOT PASSED")

    if passed:
        st.success("Congratulations! You passed the quiz!")
    else:
        st.error(f"You did not reach the minimum pass score of {min_pass}.")

    lang_completion = get_language_name(
        st.session_state.get("course_language", "en"))
    if "completion_message_for_session" not in st.session_state:
        summary = _build_completion_summary(
            questions, quiz_cfg, answers, total_earned, total_possible, passed
        )
        text_out = ""
        if api_key:
            comp_user = build_completion_user_prompt(summary, lang_completion)
            try:
                with st.spinner("Generating closing message..."):
                    raw_c, m_c, in_c, out_c, dur_c = call_openrouter(
                        QUIZ_COMPLETION_SYSTEM_PROMPT,
                        comp_user,
                        api_key,
                        completion_model,
                        max_tokens=1200,
                        temperature=0.6,
                    )
                _log_api_call(
                    phase="completion_message",
                    system_prompt=QUIZ_COMPLETION_SYSTEM_PROMPT,
                    user_prompt=comp_user,
                    raw_response=raw_c,
                    parsed_response=None,
                    model=m_c,
                    input_tokens=in_c,
                    output_tokens=out_c,
                    duration_ms=dur_c,
                )
                text_out = (raw_c or "").strip()
            except Exception as e:
                st.caption(f"Completion message skipped: {e}")
        st.session_state.completion_message_for_session = text_out

    cm = (st.session_state.completion_message_for_session or "").strip()
    if cm:
        st.info(cm)

    st.divider()

    for i, q in enumerate(questions):
        ans = answers.get(i, {})
        status = "Passed" if ans.get("passed") else (
            "Skipped" if ans.get("skipped") else "Not passed")
        icon = {"Passed": "✅", "Skipped": "⏭️",
                "Not passed": "❌"}.get(status, "")

        with st.expander(f"{icon} Q{i+1}: {q['question'][:80]}... — {status} ({ans.get('best_score', 0):.1f}/{q['score']})"):
            st.markdown(f"**Correct answers:** {', '.join(q['answers'])}")
            st.markdown(f"**Attempts:** {ans.get('attempts', 0)}")
            if ans.get("conversation"):
                st.caption("Conversation:")
                for msg in ans["conversation"]:
                    if msg["role"] == "user":
                        st.chat_message("user").write(msg["content"])
                    else:
                        st.chat_message("assistant").write(msg["content"])

    st.divider()

    # Logs download (also in sidebar, but convenient here too)
    if st.session_state.api_logs:
        st.subheader("API Logs")
        st.caption(
            f"{len(st.session_state.api_logs)} API call(s) logged during this session.")
        logs_json = json.dumps(st.session_state.api_logs,
                               ensure_ascii=False, indent=2)
        st.download_button(
            "Download full pipeline logs (JSON)",
            data=logs_json,
            file_name=f"quiz_pipeline_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="results_download_logs",
        )

        with st.expander("Preview logs"):
            for i, log in enumerate(st.session_state.api_logs):
                st.markdown(
                    f"**{i+1}. [{log['phase']}]** — {log['timestamp']} — {log['usage']['input_tokens']}in/{log['usage']['output_tokens']}out — {log['duration_ms']}ms")

    if st.button("Start new quiz"):
        for k in list(st.session_state.keys()):
            if k not in ("api_logs",):
                del st.session_state[k]
        _init_state()
        st.rerun()


if __name__ == "__main__":
    main()
