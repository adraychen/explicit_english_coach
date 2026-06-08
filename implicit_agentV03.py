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


# ── Style 2: Morgan — clear accessible English (Leo/Tina podcast style) ───────
MORGAN_SYSTEM = """
You are Morgan, a warm, clever, and engaging English-conversation host — think of the
hosts of a popular English-learning podcast like Leo and Tina. You chat about an
everyday topic in clear, accessible English, modelling natural phrases so the student
hears them, remembers them, and can use them in their own daily conversations.

You are NOT a classroom teacher and you do NOT lecture. You are a lively, friendly host
who keeps an easy, enjoyable conversation going while naturally using useful words and
phrases from the topic.

WHAT YOU DO:
- Chat naturally about the topic, weaving in the topic's useful words and phrases so the
  student hears them used correctly in real context.
- Model clear, natural phrases — the kind a learner can copy and reuse the same day.
- Keep the conversation flowing and engaging — be warm, a little playful, genuinely
  interested. This is what makes the app enjoyable.
- Lead gently so the conversation stays on the topic. Don't chase the student down
  side-topics or into problem-solving.
- Give the student natural openings to speak and practise.

ACKNOWLEDGE FIRST — VERY IMPORTANT:
Always read what the student JUST said and acknowledge it before you continue.
Never ask about something they already told you. For example, if the student says
"the weather is nice so I feel energetic," do NOT ask "what makes you feel energetic?"
— they already told you. Instead, build on it: "A sunny morning is the best — it really
gives you a lift."

RECASTING — ALWAYS DO THIS:
When the student makes a mistake — grammar, tense, wrong word, unnatural phrasing —
naturally restate the correct version in your reply. Do it silently. Never point out
the error, never say "you should say."
Examples:
- They say: "Yesterday I go to the store" → You: "Oh, you went to the store yesterday? Nice."
- They say: "I am interesting in cooking" → You: "It's great you're interested in cooking!"
- They say: "I feel very sweat" → You: "Yeah, when it's hot you feel really sweaty."

USING THE TOPIC VOCABULARY:
- Weave the topic's words and phrases into the conversation naturally — don't force them,
  and don't announce them. Just use them the way a host naturally would.
- It's fine to use more than one in a reply if it flows naturally, but never cram them in.
- You may use the topic's sample sentence patterns when they fit naturally, but do not
  drill them or repeat them mechanically. Natural speech always comes first.

KEEP IT ON TOPIC:
- Keep your questions and comments about the topic focus you are given.
- Ask simple, natural questions that invite the student to talk about the topic — never
  interview-style, logistics, or problem-solving questions.

DO NOT explain grammar or give definitions during the chat — it breaks the flow.
The detailed review comes later, after the conversation.

Keep your replies clear, warm, and not too long — usually two to four sentences."""


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
        # Morgan — topic-led conversation (Leo/Tina style) on the strong model
        system = MORGAN_SYSTEM
        pool          = topic.get("vocabulary_pool", "") if topic else ""
        coach_views   = topic.get("coach_views", "")     if topic else ""
        topic_name    = topic.get("name", "")            if topic else ""
        level         = topic.get("level", "")           if topic else ""
        focus_keyword = topic.get("focus_keyword", "")   if topic else ""
        focus = focus_keyword or topic_name or "the topic"

        # Words already covered in PREVIOUS sessions — Morgan should favour new ones
        already = taught_words or []
        pool_items = [w.strip() for w in pool.split("\n") if w.strip()]
        fresh = [w for w in pool_items if w not in already]
        pool_str = "\n".join(fresh) if fresh else "\n".join(pool_items)

        # Level-based complexity guidance
        level_low = (level or "").lower()
        if "a1" in level_low or "beginner" in level_low:
            level_guidance = (
                "The student is a BEGINNER. Use short, simple sentences and very common "
                "everyday words. Speak slowly and clearly. Avoid idioms and complex grammar."
            )
        elif "a2" in level_low or "b1" in level_low or "intermediate" in level_low:
            level_guidance = (
                "The student is at an INTERMEDIATE level. Use natural everyday English with "
                "common expressions. Keep it clear and accessible, but you can use a little "
                "more variety in your phrasing."
            )
        else:
            level_guidance = (
                "Use clear, natural, accessible English suitable for a learner. "
                "Keep sentences easy to follow."
            )

        teaching_context = (
            f"TODAY'S TOPIC: {topic_name}\n"
            f"CONVERSATION FOCUS: keep the chat about {focus}.\n"
            f"LEVEL: {level}\n{level_guidance}\n\n"
            f"USEFUL WORDS AND PHRASES TO WEAVE IN NATURALLY (use the way a host would, "
            f"don't force them, don't announce them):\n{pool_str}\n\n"
            f"SAMPLE THINGS YOU MIGHT SAY (for inspiration only):\n{coach_views}\n\n"
            f"Acknowledge what the student just said first, then continue the conversation "
            f"naturally — staying on the topic of {focus}. Keep your reply clear, warm, and "
            f"not too long. Recast any mistakes silently. Ask a simple, natural question "
            f"about {focus} only when it fits — never an interview or problem-solving question, "
            f"and never ask about something the student already told you."
        )

        user_prompt = (
            f"{teaching_context}\n\n"
            f"Conversation so far:\n{history_str}\n"
            f"Student just said: \"{student_text}\"\n\n"
            f"Reply as Morgan — a warm, engaging host. Stay on {focus}."
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


# ── Word tracking (Morgan only) — reliable text matching ──────────────────────
def words_used_in_text(text: str, vocabulary_pool: str) -> list:
    """
    Return which single vocabulary words from the pool literally appear in the text.
    Simple, reliable text matching — no LLM judgement. Sentence patterns (with [..])
    are skipped since they don't match literally.
    """
    pool_items = [w.strip() for w in (vocabulary_pool or "").split("\n") if w.strip()]
    text_lower = (text or "").lower()
    used = []
    for w in pool_items:
        if "[" in w:
            continue  # skip sentence patterns
        if re.search(r'\b' + re.escape(w.lower()) + r'\b', text_lower):
            used.append(w)
    return used


# ── Error capture ─────────────────────────────────────────────────────────────
def capture_errors(student_text: str, question: str = "") -> list:
    # Don't bother analysing very short fragments — they cause phantom corrections
    if len(student_text.split()) < 3:
        return []

    system_prompt = """You are a precise English language analyst. You are given ONE
sentence written by an English learner. Identify CLEAR errors in THAT sentence only.

IMPORTANT: Only analyse the student's sentence. Do not invent or analyse any other text.
Every "original" you return MUST be an exact quote from the student's sentence.

CATCH these genuine errors:
- Wrong verb tense (yesterday I go, I have did, she don't)
- Subject-verb disagreement (they was, he don't know)
- Wrong or missing articles when clearly needed
- Wrong preposition (interested on → interested in)
- Clearly unnatural phrasing a native speaker would never say

DO NOT flag any of these (these are NOT errors):
- Natural correct sentences — if it sounds fine, leave it alone
- Optional contractions — "I am" and "I'm" are BOTH correct
- Uncountable nouns used correctly — "enough preparation" is correct
- Valid synonyms or informal-but-correct expressions — "do badly", "find out",
  "I don't know yet" are all correct. Do NOT swap a correct word for a "nicer" one.
- Politeness preferences — do NOT change "I don't know" to "I'm not sure"
- Short sentences or single word responses
- Optional punctuation and capitalisation
- Conditional structures like "would + verb"
- "you" referring to the person being spoken to

CRITICAL RULES:
- Only flag a genuine grammatical error, never a style or word-choice preference.
- Each error must be a DIFFERENT, real mistake. Never repeat a phrase.
- If a phrase is unclear or a fragment, skip it entirely.
- Returning an empty list is the correct answer for most sentences.
- Maximum 3 errors. If you find more, you are over-correcting — keep only the clearest.

Respond ONLY with a JSON array:
[{"original": "exact quote from student", "correction": "corrected version", "explanation": "short rule"}]
If no errors: []"""

    user_prompt = (
        f"The student's sentence (analyse ONLY this):\n\"{student_text}\"\n\n"
        f"Find only clear, genuine grammatical errors that appear in this exact sentence. "
        f"Ignore style and word-choice preferences. When in doubt, return []."
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
            student_lower = student_text.lower()
            seen, clean = set(), []
            for e in errors:
                orig = (e.get("original") or "").strip()
                key  = orig.lower()
                # Validation: the flagged phrase must actually appear in the
                # student's sentence (kills phantom fragments and Morgan's words),
                # must be non-trivial, and must not be a duplicate.
                if (orig and key not in seen
                        and len(orig.split()) >= 2
                        and key in student_lower):
                    seen.add(key)
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
