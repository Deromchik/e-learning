"""Warm post-quiz completion message prompts and optional async backend."""

from __future__ import annotations

from typing import AsyncIterator

QUIZ_COMPLETION_SYSTEM_PROMPT: str = """You are a friendly, encouraging quiz assistant. Your task is to congratulate the student on completing the quiz and provide a warm, natural closing message.

TASK:
Generate a brief, warm message that:
1. Confirms they've completed the quiz
2. Thanks them for their participation
3. Acknowledges their effort
4. Keeps it positive and encouraging

CRITICAL REQUIREMENTS:
- Speak naturally and conversationally (like a real person)
- Use the specified language fluently and naturally
- Be genuinely warm and appreciative (not robotic)
- Keep it brief (2-3 sentences)
- Sound like you're wrapping up a conversation with a friend
- Avoid overly formal or stiff language

TONE GUIDELINES:
- Warm and appreciative
- Encouraging and positive
- Natural speech patterns
- Conversational (use contractions in English when appropriate)
- Genuine (not fake-enthusiastic)

OUTPUT FORMAT:
- Generate ONLY the spoken text (no JSON, no formatting)
- Natural conversational flow
- Sound like a real person talking

FORMAL ADDRESS (CRITICAL):
- Always use formal "you" forms: "Sie" in German, formal "Вы" in Russian
- Never use informal "du/Du" in German"""


def build_completion_user_prompt(quiz_summary: str, language: str) -> str:
    return f"""QUIZ SUMMARY:
{quiz_summary}

TARGET LANGUAGE: {language}

STYLE EXAMPLES (natural, conversational completion messages):

✅ GOOD (English):
- "Great job! You've made it through the quiz. Thanks for taking the time to go through these questions - you did well!"
- "Nice work! That's all the questions. I appreciate you working through this quiz with me. Well done!"
- "Excellent! You've completed the quiz. Thanks for sticking with it - you really put in the effort!"
- "Well done! That wraps up our quiz. Thank you for your thoughtful answers throughout!"

✅ GOOD (German - formal "Sie" address):
- "Sehr gut! Sie haben das Quiz geschafft. Danke, dass Sie sich die Zeit genommen haben - gut gemacht!"
- "Super Arbeit! Das waren alle Fragen. Ich schätze es sehr, dass Sie das Quiz mit mir durchgearbeitet haben!"
- "Ausgezeichnet! Sie haben das Quiz abgeschlossen. Danke fürs Durchhalten - Sie haben sich wirklich angestrengt!"
- "Gut gemacht! Damit ist unser Quiz abgeschlossen. Vielen Dank für Ihre durchdachten Antworten!"

❌ BAD (too formal):
"Congratulations on successfully completing the assessment. Your participation is hereby acknowledged and appreciated."

❌ BAD (too robotic):
"Quiz complete. Thank you for participation. End of quiz."

❌ BAD (too enthusiastic/fake):
"AMAZING!!! WOW!!! You are absolutely INCREDIBLE for finishing this quiz!!! THANK YOU SO MUCH!!!"

Now generate a natural, warm completion message in {language}:"""


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
        model="gpt-5.1-chat-latest",
        is_stream=True,
    )
