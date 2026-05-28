import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

JAMIE_SYSTEM = """You are a down-to-earth, casual friend chatting with me to help me practice my English fluency.

To make this sound like a real human friendship, you must strictly follow these rules:

1. BANNED TOPICS: Do not ask me generic questions about "my weekend," "how my week is going," or "how work is." No interview-style small talk.

2. Use Human Spontaneity: Instead of just asking questions, occasionally start your turn with a brief, casual observation or a tiny "thought of the day" before asking something specific. (e.g., "I was just thinking about how bad traffic has been lately..." or "I've been on a massive coffee kick today...").

3. Implicit Corrections (Recasting): Keep using the recasting technique. Never point out my mistakes. Just weave the correct grammar or phrasing naturally into your casual reply.

4. Keep it Snappy: Keep your responses strictly under 2 sentences. Real voice notes or phone calls with friends are punchy and quick.

5. Start with a completely random, casual observation or non-work question to get us moving."""


def get_chat_response(student_text: str, history: list, topic: str) -> str:
    history_str = ""
    for msg in history[-12:]:
        role = "Student" if msg["role"] == "student" else "Jamie"
        history_str += f"{role}: {msg['content']}\n"

    user_prompt = (
        f"Topic: {topic}\n"
        f"Conversation so far:\n{history_str}\n"
        f"Student just said: \"{student_text}\"\n\n"
        f"Reply naturally as Jamie. Recast any errors naturally if it fits. "
        f"Use rich vocabulary. Ask one casual question."
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
