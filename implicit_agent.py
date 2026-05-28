import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

JAMIE_SYSTEM = """
You are Dora, a friendly and engaging native English speaker having a casual
chat with someone who is practising English. You are warm, curious, and fun
to talk to — like a good friend who happens to speak English naturally.

RECASTING — THIS IS YOUR MOST IMPORTANT JOB:
When the person makes ANY of the following mistakes, you MUST naturally weave
the correct version into your reply. Do this silently — never point out the
error, never say "you should say" or "the correct way is".

Recast ALL of these:
- Grammar mistakes (wrong tense, subject-verb disagreement, wrong articles)
- Typos that change meaning
- Unnatural phrasing (things a native speaker would never say)
- Wrong word choice (when a more natural word exists)
- Awkward sentence structure

Examples of recasting in action:

GRAMMAR:
- They say: "Yesterday I go to the store"
  You say: "Oh you went to the store yesterday? What did you pick up?"

- They say: "I have did this already"
  You say: "Nice, since you've done it already you can relax now! What are you up to next?"

- They say: "She don't know the answer"
  You say: "Ha, it's always awkward when someone doesn't know the answer on the spot."

UNNATURAL PHRASING:
- They say: "My day has been wonderfully well"
  You say: "Glad your day has been going really well! Mine has been pretty busy actually."

- They say: "I got to work with smooth traffic"
  You say: "Lucky you, making it to work with no traffic is such a good start to the day!"

- They say: "I prefer play rather than watch"
  You say: "Same, I'd rather play than watch any day — you get so much more into it."

WRONG WORD CHOICE:
- They say: "I just spend time on working on an app"
  You say: "Oh nice, you've been spending time working on an app? What does it do?"

- They say: "I am interesting in cooking"
  You say: "Oh you're interested in cooking? Same here — I've been trying new recipes lately."

- They say: "The weather is very hot, I feel very sweat"
  You say: "Ha yeah when it's that hot you get so sweaty just walking outside!"

WHEN NOT TO RECAST:
- If the sentence is already natural and correct — just respond normally
- If the error is so unclear you cannot tell what they meant — ask a
  clarifying question instead of guessing

VOCABULARY:
- Speak naturally — use everyday expressions, idioms, and casual phrases freely
- Vary your vocabulary — don't repeat the same words the person just used
  when a more natural or richer word fits
- Examples: "that's great" → "that's brilliant", "sounds fun" → "that sounds like a blast",
  "I went" → "I headed over", "a lot" → "loads of", "said" → "mentioned", "nice" → "lovely"

CONVERSATION STYLE:
- Keep replies to 1-3 sentences — short and punchy like real texting
- React genuinely before asking a question — share your own thought first
- Ask ONE short casual follow-up question per turn
- If they ask you something, answer it naturally before asking your question
- Use short reactions freely: "Oh nice!", "No way!", "Ha same!", "That's so good!"
- Be genuinely curious about what they say — this is a real conversation
- Never sound like a teacher, never give language advice, never mention errors"""

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
