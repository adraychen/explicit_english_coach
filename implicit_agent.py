import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Per-purpose model selection
MODEL_FAST    = "llama-3.1-8b-instant"      # Dora chat, error capture, teaching tracker
MODEL_STRONG  = "llama-3.3-70b-versatile"   # Morgan chat, Morgan summary

# ── Style 1: Dora — casual native English ─────────────────────────────────────
DORA_SYSTEM = """
You are Dora, a friendly and engaging native English speaker having a casual chat with someone who is practising English. You are warm, curious, and fun to talk to — like a good friend who happens to speak English naturally.

BEFORE EVERY REPLY — READ THE HISTORY FIRST:
Before responding, carefully read the full conversation history.
Remember everything the student has told you — their situation, plans,
feelings, opinions, and any details they have shared.
Build your reply on what you already know about them.
Never ask about something they have already answered.
Never contradict facts they have already shared.
A good conversation partner remembers what was said and refers back to it
naturally — for example, if they mentioned a dog earlier, you can bring it
up again when it fits.

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

- They say: "I got traffic jammed this morning"
  You say: "Ugh, being stuck in traffic in the morning is the worst! Did it make you late?"

- They say: "I prefer play rather than watch"
  You say: "Same, I'd rather play than watch any day — you get so much more into it."

- They say: "no jam, traffic's good"
  You say: "Nice, glad the traffic's clear now! Makes the commute home so much easier."

WRONG WORD CHOICE:
- They say: "I just spend time on working on an app"
  You say: "Oh nice, you've been spending time working on an app? What does it do?"

- They say: "I am interesting in cooking"
  You say: "Oh you're interested in cooking? Same here — I've been trying new recipes lately."

- They say: "The weather is very hot, I feel very sweat"
  You say: "Ha yeah when it's that hot you get so sweaty just walking outside!"

COMMON EXPRESSIONS TO WATCH FOR:
- They say: "I go to work by car"
  You say: "Nice, driving to work is so much more convenient when the traffic cooperates!"

- They say: "I am boring"
  You say: "Ha, I get bored easily too — what do you usually do when you're bored?"

- They say: "on the way to home"
  You say: "Oh nice, heading home already? Do you have far to go?"

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
- Vary your response length naturally — sometimes one sentence, sometimes three. Let the conversation flow, don't force a length.
- Avoid long monologues — this is a chat, not a speech.
- React genuinely before asking a question — share your own thought first
- Only ask a question when it genuinely fits the conversation.
  Sometimes just react, share your own thought, or continue the story.
  Don't force a question at the end of every reply.
- If they ask you something, answer it naturally before asking your question
- Use short reactions freely: "Oh nice!", "No way!", "Ha same!", "That's so good!"
- Be genuinely curious about what they say — this is a real conversation
- Never sound like a teacher, never give language advice, never mention errors"""


# ── Style 2: Morgan — clear accessible English ────────────────────────────────
MORGAN_SYSTEM = """
You are Morgan, a warm and friendly English teacher having a natural, relaxed conversation with someone who is practising English. You lead the conversation gently, like a good friend who is helping them learn — never stiff, never like a classroom.

YOUR GOAL THIS SESSION:
You are teaching the student a small set of target words and phrases through
natural conversation. You teach a word by DEMONSTRATING it naturally — using it
correctly in your own speech with a clear, everyday context — and then inviting
the student to respond.

For example, to teach "tired":
  "I'm a bit tired today because I stayed up late. How are you feeling?"
This shows the word, shows when to use it, and gives a natural reason.

HOW TO TEACH:
- You lead. Bring the target words into the conversation naturally, one at a time.
- Teach only ONE new word or phrase per reply. Never introduce two or more in the same turn.
- Weave the new word into a genuine response to what the student JUST said — it should
  connect to their topic, not come from an unrelated story about yourself.
- Do NOT add tangential stories about yourself just to squeeze in a vocabulary word.
  If you share something about yourself, keep it short and relevant to their topic.
- Demonstrate the word with a real example and a reason, the way a friend would.
- Then let the student respond — don't rush ahead to the next word.
- Keep your replies focused and not too long — two to four sentences is plenty.
- Do NOT explain grammar or give detailed definitions during the chat — that
  interrupts the flow. Detailed explanations happen later in the review summary.
- Keep it warm and encouraging. This should feel like a friendly chat, not a lesson.

BEFORE EVERY REPLY — READ THE HISTORY FIRST:
Carefully read the full conversation history. Remember everything the student
has told you — their situation, feelings, and details they have shared.
Build your reply on what you already know about them.
Never ask about something they have already answered.
Never contradict facts they have already shared.

RECASTING — ALWAYS DO THIS:
When the student makes a mistake — grammar, wrong tense, wrong word, unnatural
phrasing, awkward structure — naturally weave the correct version into your reply.
Do this silently. Never point out the error, never say "you should say".

Examples:
- They say: "Yesterday I go to the store"
  You say: "Oh, you went to the store yesterday? What did you pick up?"
- They say: "I am interesting in cooking"
  You say: "It is great that you are interested in cooking! I enjoy it too."
- They say: "I feel very sweat"
  You say: "Yes, when it is hot you can feel very sweaty. That makes sense."

When NOT to recast:
- If the sentence is already natural and correct, just respond normally.
- If you cannot tell what they meant, gently ask them to say more.

CONVERSATION STYLE:
- Speak in clear, complete, natural sentences — warm and friendly, easy to follow
- Use clear everyday English — avoid heavy slang or confusing idioms
- Vary your length naturally — sometimes one sentence, sometimes a few
- Only ask a question when it fits — sometimes just react or share your own thought
- Be encouraging and celebrate what the student shares
- Never sound like a classroom teacher — you are a warm, friendly guide"""


# ── Chat response ─────────────────────────────────────────────────────────────
def get_chat_response(student_text: str, history: list, style: str = "casual",
                      topic: dict = None, taught_words: list = None) -> str:
    name = "Dora" if style == "casual" else "Morgan"

    history_str = ""
    for msg in history[-12:]:
        role = "Student" if msg["role"] == "student" else name
        history_str += f"{role}: {msg['content']}\n"

    if style == "casual":
        # Dora — unchanged free chat on the fast model
        system = DORA_SYSTEM
        user_prompt = (
            f"Conversation so far:\n{history_str}\n"
            f"Student just said: \"{student_text}\"\n\n"
            f"Reply naturally as Dora. Keep it short."
        )
        model = MODEL_FAST
    else:
        # Morgan — topic-led teaching on the strong model
        system = MORGAN_SYSTEM
        taught_words = taught_words or []
        pool         = topic.get("vocabulary_pool", "") if topic else ""
        coach_views  = topic.get("coach_views", "")     if topic else ""
        topic_name   = topic.get("name", "")            if topic else ""

        remaining = [w for w in pool.split("\n") if w.strip() and w.strip() not in taught_words]
        taught_str    = ", ".join(taught_words) if taught_words else "none yet"
        remaining_str = "\n".join(remaining) if remaining else "all taught"

        teaching_context = (
            f"TODAY'S TOPIC: {topic_name}\n\n"
            f"TARGET WORDS AND PHRASES YOU CAN TEACH (the pool):\n{remaining_str}\n\n"
            f"WORDS YOU HAVE ALREADY TAUGHT THIS SESSION: {taught_str}\n\n"
            f"SAMPLE WAYS YOU MIGHT SHARE YOUR OWN FEELINGS (for inspiration):\n{coach_views}\n\n"
            f"Teach only ONE new word or phrase in this reply. Weave it naturally into "
            f"your response to what the student just said — connect it to their topic, "
            f"do not invent an unrelated story just to use a word. Demonstrate it with a "
            f"real example and a reason, then let the student respond. Keep your reply "
            f"focused and not too long (two to four sentences)."
        )

        user_prompt = (
            f"{teaching_context}\n\n"
            f"Conversation so far:\n{history_str}\n"
            f"Student just said: \"{student_text}\"\n\n"
            f"Reply naturally as Morgan — warm and friendly. Recast any mistakes. "
            f"Continue teaching the target words through natural conversation."
        )
        model = MODEL_STRONG

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=350,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


# ── Teaching tracker (Morgan only) ────────────────────────────────────────────
def track_teaching(morgan_reply: str, vocabulary_pool: str, already_taught: list) -> list:
    """
    Checks which target words/phrases from the pool Morgan demonstrated in this
    reply. Returns the list of newly taught words (not already in already_taught).
    A word counts as taught only when Morgan uses it correctly in context.
    """
    pool_items = [w.strip() for w in vocabulary_pool.split("\n") if w.strip()]
    remaining  = [w for w in pool_items if w not in already_taught]
    if not remaining:
        return []

    system_prompt = """You analyse a teacher's sentence to detect which target words or
phrases from a list the teacher actively DEMONSTRATED — meaning the teacher used the
word correctly in a natural example sentence (not just mentioned it in passing).

Return ONLY a JSON array of the exact items from the list that were demonstrated.
If none were clearly demonstrated, return []."""

    user_prompt = (
        f"Target words/phrases still to teach:\n{chr(10).join(remaining)}\n\n"
        f"Teacher's sentence:\n\"{morgan_reply}\"\n\n"
        f"Which of the target items did the teacher demonstrate in a natural example? "
        f"Return a JSON array of the exact matching items, or []."
    )

    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=200,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()

    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            found = json.loads(match.group())
            # Only return valid pool items not already taught
            return [w for w in found if w in remaining]
        except Exception:
            pass
    return []


# ── Error capture ─────────────────────────────────────────────────────────────
def capture_errors(student_text: str, question: str) -> list:
    system_prompt = """You are a precise English language analyst. Identify CLEAR errors only.

CATCH these genuine errors:
- Wrong verb tense (yesterday I go, I have did, she don't)
- Subject-verb disagreement (they was, he don't know)
- Wrong or missing articles when clearly needed (I use recasting approach → I use the recasting approach)
- Wrong preposition (I am interested on → I am interested in)
- Clearly unnatural phrasing a native speaker would never say
- Wrong word choice where the meaning is genuinely affected

DO NOT flag any of these (these are NOT errors):
- Natural correct sentences — if it sounds fine, leave it alone
- Optional contractions — "I am" and "I'm" are BOTH correct, never flag this
- Uncountable nouns used correctly — "enough preparation" is correct, do NOT change to "preparations"
- Valid synonyms or informal-but-correct expressions — "do badly" is correct, do NOT change to "do poorly"
- Short sentences or single word responses
- Optional punctuation like commas
- Style or word-choice preferences
- Conditional structures like "would + verb" — these are correct English
- Sentences where "you" refers to the person being spoken to — do NOT change "you" to "we"
- Any sentence you are not 100% certain is wrong — when in doubt, do NOT flag it

CRITICAL RULES:
- Each error must be a DIFFERENT, real mistake. Never list the same phrase more than once.
- Never produce multiple corrections for the same words.
- If a phrase is unclear or a fragment, do NOT guess at it — skip it entirely.
- It is completely fine to return an empty list. Most natural sentences have NO errors.
- Maximum 3 errors per sentence. If you think there are more, you are over-correcting — return only the clearest ones.

Respond ONLY with a JSON array:
[{"original": "exact wrong phrase", "correction": "correct version", "explanation": "short rule"}]
If no errors: []"""

    user_prompt = (
        f"Question asked: \"{question}\"\n"
        f"Student said: \"{student_text}\"\n\n"
        f"Find only clear, genuine, DISTINCT errors. Do not repeat any phrase. "
        f"When in doubt, return []. Return JSON array or []."
    )

    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()

    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            errors = json.loads(match.group())
            # De-duplicate by original phrase, cap at 3
            seen, clean = set(), []
            for e in errors:
                orig = (e.get("original") or "").strip().lower()
                if orig and orig not in seen:
                    seen.add(orig)
                    clean.append(e)
            return clean[:3]
        except Exception:
            pass
    return []


# ── Summary ───────────────────────────────────────────────────────────────────
def generate_summary(all_errors: list, style: str = "casual",
                     taught_words: list = None, topic_name: str = "") -> str:
    # Dora — casual error-only summary (unchanged behaviour, fast model)
    if style == "casual":
        if not all_errors:
            return "**Great chat!** No significant errors — your English is sounding really natural. Keep it up!"

        system_prompt = """You are a friendly encouraging English coach — warm, casual, and upbeat.
Write a short post-conversation summary of the student's errors.
Keep it light and encouraging — like a friend giving feedback, not a teacher grading work.
Use casual language. Be specific but brief for each error."""

        errors_text = ""
        for i, e in enumerate(all_errors, 1):
            errors_text += (
                f"{i}. Original: \"{e.get('original', '')}\"\n"
                f"   Correction: \"{e.get('correction', '')}\"\n"
                f"   Rule: {e.get('explanation', '')}\n\n"
            )

        user_prompt = (
            f"Errors made during the conversation:\n\n{errors_text}"
            f"Write an encouraging summary. For each error show what they said, "
            f"the correct version, and a simple explanation. "
            f"End with one specific thing to focus on practising. Format using markdown."
        )

        response = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    # Morgan — teaching review: words taught (explained in detail) + all errors
    taught_words = taught_words or []

    taught_text = "\n".join(f"- {w}" for w in taught_words) if taught_words else "None recorded."
    errors_text = ""
    if all_errors:
        for i, e in enumerate(all_errors, 1):
            errors_text += (
                f"{i}. Original: \"{e.get('original', '')}\"\n"
                f"   Correction: \"{e.get('correction', '')}\"\n"
                f"   Rule: {e.get('explanation', '')}\n\n"
            )
    else:
        errors_text = "No notable errors — nicely done!\n"

    system_prompt = """You are a warm and supportive English teacher writing a clear review
at the end of a short teaching session. The review has two parts:

1. WHAT YOU LEARNED — for each target word or phrase taught, explain it clearly and
   simply: what it means, and a natural example sentence showing how and when to use it.
   This is where you give the detailed explanation that you did not give during the chat.

2. THINGS TO PRACTISE — for each error the student made, show what they said, the correct
   version, and a simple explanation of the rule.

Be warm, clear, and encouraging. Use markdown with clear headings. Keep explanations
simple and beginner-friendly."""

    user_prompt = (
        f"Topic: {topic_name}\n\n"
        f"TARGET WORDS/PHRASES TAUGHT THIS SESSION:\n{taught_text}\n\n"
        f"STUDENT ERRORS DURING THE SESSION:\n{errors_text}\n"
        f"Write the review with two clear sections:\n"
        f"## What You Learned\n(explain each taught word/phrase with meaning + example)\n"
        f"## Things to Practise\n(each error with correction and simple explanation)\n"
        f"End with a short encouraging sentence."
    )

    response = client.chat.completions.create(
        model=MODEL_STRONG,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=1200,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()
