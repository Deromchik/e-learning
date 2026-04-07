"""

Для каждого промпта указано:
- текущая модель (значение по умолчанию и способ изменения),
- параметры вызова модели (temperature, max_tokens, reasoning и т. д.),
- способ передачи данных в API (что отправляется в `system`, а что — в `user`).
"""

PROMPT_CONFIGS = {
    # ------------------------------------------------------------------
    # 1. Генерация тестовых вопросов (SHORT ANSWER / REASONING)
    # ------------------------------------------------------------------
    "generate_questions": {
        "description": (
            "Генерация GIFT-quiz вопросов (краткие ответы или reasoning) "
            "на основе Gift-текста, дополнительных материалов и предыдущих уроков."
        ),
        "model": {
            "current_default": "google/gemini-2.5-pro",
            "api_provider": "OpenRouter (OpenAI-совместимый /chat/completions)",
        },
        "parameters": {
            "max_tokens": 10000,
            "temperature": 0.3,
            "top_p": 0.8,
            "retries": 3,
            "extra_body": None,
        },
        "openrouter_call_shape": {
            "messages": [
                {
                    "role": "system",
                    "content_source": "build_system_prompt(course_language)",
                    "content_details": "PROMPT_SYSTEM_TEMPLATE: переменные language_name, course_language",
                },
                {
                    "role": "user",
                    "content_source": "build_giftquiz_*_questions_prompt(payload, course_language, difficulty_level)",
                    "content_details": (
                        "payload['data'].gift, payload['data'].extracted_text, "
                        "payload['data'].lesson_other_topics, payload['data'].previous_lesson_summaries, "
                        "difficulty_level"
                    ),
                },
            ],
        },
    },

    # ------------------------------------------------------------------
    # 2. Валидация ответа пользователя
    # ------------------------------------------------------------------
    "validate_answer": {
        "description": (
            "Оценка ответа студента на отдельный вопрос: "
            "расчёт validation_score, answer_score, user_intent и "
            "генерация подсказки без спойлеров."
        ),
        "model": {
            "current_default": "openai/gpt-4o",
            "api_provider": "OpenRouter (OpenAI-совместимый /chat/completions)",
        },
        "parameters": {
            "max_tokens": 2000,
            "temperature": 0.3,
            "top_p": 0.8,
            "retries": 3,
            "extra_body": {
                "reasoning": {"effort": "medium"},
            },
        },
        "openrouter_call_shape": {
            "messages": [
                {
                    "role": "system",
                    "content_source": "build_answer_validation_system_prompt(validation_instruction)",
                    "content_details": "quiz_answer_validation.QUIZ_ANSWER_VALIDATION_SYSTEM_PROMPT_BASE + DIFFICULTY_LEVELS[level].validation_instruction",
                },
                {
                    "role": "user",
                    "content_source": "build_answer_validation_user_prompt(...)",
                    "content_details": "quiz_question, correct_answers, user_answer, conversation_history, max_possible_score, language",
                },
            ],
        },
    },

    # ------------------------------------------------------------------
    # 3. Розмовна подача питання (rephraser)
    # ------------------------------------------------------------------
    "rephrase_question": {
        "description": (
            "Перефразування питання квізу під розмовну подачу; інтро до квізу на першому питанні."
        ),
        "model": {
            "current_default": "openai/gpt-5.1-chat-latest",
            "api_provider": "OpenRouter",
            "env": "OPENROUTER_REPHRASE_MODEL",
        },
        "parameters": {
            "max_tokens": 2000,
            "temperature": 0.85,
            "top_p": 0.9,
        },
        "openrouter_call_shape": {
            "messages": [
                {
                    "role": "system",
                    "content_source": "quiz_question_rephraser.QUIZ_REPHRASER_SYSTEM_PROMPT",
                },
                {
                    "role": "user",
                    "content_source": "build_rephraser_user_prompt(...)",
                },
            ],
        },
    },

    # ------------------------------------------------------------------
    # 4. Тьюторський follow-up після невдалої валідації
    # ------------------------------------------------------------------
    "followup_after_fail": {
        "description": (
            "Короткий дружній follow-up у діалозі після невдалої спроби; не дублює validation_error дослівно."
        ),
        "model": {
            "current_default": "openai/gpt-5.1-chat-latest",
            "env": "OPENROUTER_FOLLOWUP_MODEL",
        },
        "parameters": {
            "max_tokens": 2000,
            "temperature": 0.85,
            "top_p": 0.9,
        },
        "openrouter_call_shape": {
            "messages": [
                {
                    "role": "system",
                    "content_source": "quiz_followup_question.QUIZ_FOLLOWUP_SYSTEM_PROMPT",
                },
                {
                    "role": "user",
                    "content_source": "build_followup_user_prompt(...)",
                },
            ],
        },
    },

    # ------------------------------------------------------------------
    # 5. Повідомлення після завершення квізу
    # ------------------------------------------------------------------
    "completion_message": {
        "description": "Коротке тепле завершення діалогу після екрану результатів.",
        "model": {
            "current_default": "openai/gpt-5.1-chat-latest",
            "env": "OPENROUTER_COMPLETION_MODEL",
        },
        "parameters": {
            "max_tokens": 1200,
            "temperature": 0.6,
        },
        "openrouter_call_shape": {
            "messages": [
                {
                    "role": "system",
                    "content_source": "quiz_completion_message.QUIZ_COMPLETION_SYSTEM_PROMPT",
                },
                {
                    "role": "user",
                    "content_source": "build_completion_user_prompt(quiz_summary, language)",
                },
            ],
        },
    },
}


def get_prompt_config(name: str) -> dict | None:
    """
    Повертає словник-конфіг для вказаного ключа промпту або None.
    Ключі: generate_questions, validate_answer, rephrase_question,
    followup_after_fail, completion_message.
    """
    return PROMPT_CONFIGS.get(name)
