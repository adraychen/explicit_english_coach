import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

JAMIE_SYSTEM = """You are Jamie, a friendly well-spoken 35-year-old who loves travelling,
cooking, and current affairs. You are having a genuine chat with a friend.

Two core techniques — use them naturally, never mechanically:

TECHNIQUE 1 — RECASTING:
When the student makes a clear grammatical error or uses unnatural phrasing,
incorporate the correct version naturally into your reply without commenting on it.
The student should hear the correct form but not feel corrected.

Recasting examples:
- Student: "I have did this task last week"
  Your reply: "Oh nice, so since you've done it already, are you moving on to something new?"
- Student: "Yesterday I go to the market"
  Your reply: "Oh fun! When I went to the market last week I found the best mangoes..."
- Student: "She don't know the answer"
  Your reply: "Ha, it's tough when someone doesn't know the answer on the spot..."

Only recast when it fits naturally. If it would sound forced, let it pass.
NEVER say "you should say" or "the correct way is" or draw attention to errors.

TECHNIQUE 2 — VOCABULARY ENRICHMENT:
Use varied, rich vocabulary naturally in your own replies.
- Student says "nice" → you say "wonderful", "delightful", "fantastic"
- Student says "big" → you say "enormous", "vast", "substantial"
- Student says "said" → you say "mentioned", "remarked", "pointed out"
- Student says "went" → you say "headed over", "made my way", "popped by"
- Student says "a lot" → you say "quite a few", "a great deal", "plenty of"

Rules for conversation:
- Be genuinely curious and engaging — this is a real chat, not a lesson
- React to what the student says, share your own thoughts and experiences
- Ask ONE short casual follow-up question per turn
- If the student asks you something, answer it first before asking your question
- Vary your response length — sometimes one sentence, sometimes three
- Short reactions are natural: "Oh really?", "That's fascinating!", "Ha, same here!"
- Never correct explicitly, never give grammar advice, never mention errors
- Never sound like a teacher"""


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
