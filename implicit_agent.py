import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

JAMIE_SYSTEM = """You are a casual friend chatting with me to help me practice English.

IMPORTANT - USE SIMPLE ENGLISH:
- Use simple, everyday words only
- NO idioms (e.g., "spice of life", "itching for")
- NO fancy words (e.g., "aficionado", "culinary", "remarkable", "captivating")
- NO complex phrases (e.g., "flavor profile", "thrill-seeker")
- Write like you're talking to someone learning English
- Use words like: good, nice, fun, like, want, try, go, see, eat, cool, great

Rules:
1. Keep responses to 1-2 short sentences only.

2. Implicit Corrections (Recasting): If I make a grammar mistake, use the correct version naturally in your reply. Never point out my mistakes directly.

3. Be natural and friendly, like texting a friend.

4. Ask simple follow-up questions to keep the conversation going."""


def get_chat_response(student_text: str, history: list) -> str:
    history_str = ""
    for msg in history[-12:]:
        role = "Student" if msg["role"] == "student" else "Friend"
        history_str += f"{role}: {msg['content']}\n"

    user_prompt = (
        f"Conversation so far:\n{history_str}\n"
        f"Student just said: \"{student_text}\"\n\n"
        f"Reply naturally. Keep it simple and short."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JAMIE_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=300,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def capture_errors(student_text: str, question: str) -> list:
    system_prompt = """You are a precise English language analyst. Identify CLEAR errors only.

Catch: wrong verb tense, subject-verb disagreement, wrong/missing articles,
wrong prepositions, clearly unnatural phrasing.

Do NOT flag: short sentences, style preferences, natural sentences.

Respond ONLY with a JSON array:
[{"original": "wrong phrase", "correction": "correct version", "explanation": "short rule"}]
If no errors: []"""

    user_prompt = (
        f"Question asked: \"{question}\"\n"
        f"Student said: \"{student_text}\"\n\n"
        f"Find clear errors. Return JSON array or []."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()

    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return []


def generate_summary(all_errors: list) -> str:
    if not all_errors:
        return "**Great conversation!** No significant errors were found. Your English is sounding very natural."

    system_prompt = """You are a warm encouraging English coach writing a post-conversation
summary. Present errors clearly, explain rules simply, and encourage the student.
Be warm and positive — focus on learning, not failure."""

    errors_text = ""
    for i, e in enumerate(all_errors, 1):
        errors_text += (
            f"{i}. Original: \"{e.get('original', '')}\"\n"
            f"   Correction: \"{e.get('correction', '')}\"\n"
            f"   Rule: {e.get('explanation', '')}\n\n"
        )

    user_prompt = (
        f"Errors made during the conversation:\n\n{errors_text}"
        f"Write a friendly encouraging summary. For each error show what they said, "
        f"the correct version, and a simple explanation. "
        f"End with one specific thing to focus on practising. "
        f"Format clearly using markdown."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=800,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()
