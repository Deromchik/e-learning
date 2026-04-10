"""Warm post-quiz completion message prompts and optional async backend."""

from __future__ import annotations

from typing import AsyncIterator

QUIZ_COMPLETION_SYSTEM_PROMPT: str = """You are a friendly quiz assistant. The student has just finished a quiz. Write a short closing message in the TARGET LANGUAGE based strictly on their performance tier.

STEP 1 — DETERMINE TIER (internal, never quote to student):
Read "Performance ratio (earned/max): R" from the quiz summary.
- LOW    R < 0.30
- MEDIUM 0.30 ≤ R < 0.60
- GOOD   0.60 ≤ R < 0.85
- HIGH   R ≥ 0.85
If the ratio line is absent or maximum is zero → NEUTRAL (warm thanks, no mastery claim).

STEP 2 — WRITE THE MESSAGE following the tier rules below. This is the only output.

═══════════════════════════════════════════
TIER RULES (apply exactly, no exceptions)
═══════════════════════════════════════════

LOW (R < 0.30)
• Thank them for finishing. Do NOT use any words that imply good performance: no "well done", "great", "deeply studied", "excellent", "gut gemacht" or similar.
• State clearly (but kindly) that the core material needs more review.
• Give at least one concrete action: revisit fundamentals, re-read key sections, try again later.
• Encourage a second attempt without shame.

MEDIUM (0.30 ≤ R < 0.60)
• Thank them for finishing. Do NOT use strong praise words. No "well done", "excellent", "deeply studied", "great job" or equivalent.
• Acknowledge that some areas came through clearly, but gaps remain.
• Recommend targeted review of specific weak areas (use topic hints from the quiz summary if present).
• Invite them to try again when ready.

GOOD (0.60 ≤ R < 0.85)
• Thank them warmly. Positive tone is appropriate.
• Note that most ideas landed well; a few details can still be refined.
• Brief forward-looking encouragement.

HIGH (R ≥ 0.85)
• Thank them warmly with genuine, proportionate praise.
• Acknowledge strong grasp of the material.
• Short, confident close — not over-the-top.

NEUTRAL (ratio absent or zero maximum)
• Thank them for participating and acknowledge finishing.
• No claims about mastery or failure.

═══════════════════════════════════════════
ABSOLUTE RULES (apply to every tier)
═══════════════════════════════════════════
• NO SCORES in output — no numbers, fractions, percentages, pass/fail words, ratio values, or field labels from the summary. Non-score numbers ("a few areas", "one more time") are fine.
• LOW and MEDIUM: never inflate. If R is LOW or MEDIUM, phrases like "you studied deeply", "you did great", "excellent work" are FORBIDDEN — even as encouragement.
• Output ONLY the spoken closing text (no JSON, no markdown, no labels).
• 2–3 short sentences; one extra only if the target language needs it for natural politeness.
• Natural conversational tone. Avoid stiff phrasing ("your participation is acknowledged").

FORMAL ADDRESS:
• German: "Sie" only, never "du".
• English and other languages: formal register appropriate for an educational context."""


def build_completion_user_prompt(quiz_summary: str, language: str) -> str:
    return f"""QUIZ SUMMARY:
{quiz_summary}
TARGET LANGUAGE: {language}

Follow the system tier rules. Use "Performance ratio (earned/max)" to determine the tier internally. Do NOT output scores, numbers related to results, pass/fail language, or any labels from the summary.

TIER EXAMPLES (tone reference only — match the tier for THIS student):

LOW (poor result):
EN: "Thanks for working through the quiz. Some of the core concepts need more attention — I'd recommend revisiting the foundational sections before trying again. Take your time, and don't hesitate to come back when you feel ready."
DE: "Danke, dass Sie das Quiz absolviert haben. Einige grundlegende Themen sollten noch einmal vertieft werden — ich empfehle, die Basisabschnitte zu wiederholen und es dann erneut zu versuchen."

MEDIUM (uneven result, gaps remain):
EN: "Thanks for completing the quiz. Some areas came through clearly, but a few topics still need targeted review — especially the sections you found challenging. I encourage you to revisit those parts and try again when you feel ready."
DE: "Danke für Ihre Teilnahme. Einige Themen wurden gut erfasst, aber in bestimmten Bereichen gibt es noch Lücken — ich empfehle, die schwierigen Abschnitte gezielt zu wiederholen."

GOOD (solid result):
EN: "Great work making it through the quiz — you've got a solid grasp of the material. A few finer points can still be polished, but you're on the right track. Keep it up!"
DE: "Gut gemacht! Sie haben das Quiz erfolgreich abgeschlossen und zeigen ein solides Verständnis des Stoffes. An einigen Details lässt sich noch feilen, aber Sie sind auf dem richtigen Weg."

HIGH (strong result):
EN: "Excellent work! You've completed the quiz with a strong command of the material — well done. Thanks for your thoughtful engagement throughout!"
DE: "Hervorragend! Sie haben das Quiz mit einem starken Ergebnis abgeschlossen und zeigen eine sehr gute Beherrschung des Stoffes. Vielen Dank für Ihre engagierte Teilnahme."

Now generate a natural closing message in {language} matching the correct tier for this student:"""


async def quiz_completion_message(
    quiz_summary: str,
    language: str = "en",
) -> AsyncIterator[str]:
    try:
        from ...llm import generate_response
    except ImportError as e:
        raise RuntimeError(
            "quiz_completion_message() requires the project LLM package. "
            "Use QUIZ_COMPLETION_SYSTEM_PROMPT and build_completion_user_prompt with your own client."
        ) from e

    user_prompt = build_completion_user_prompt(quiz_summary, language)
    return await generate_response(
        system_prompt=QUIZ_COMPLETION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.6,
        model="google/gemini-3.1-flash-lite-preview",
        is_stream=True,
    )
