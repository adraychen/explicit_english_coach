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
                      topic: dict = None, taught_words: list = None,
                      is_closing: bool = False) -> str:
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

        if is_closing:
            user_prompt = (
                f"{teaching_context}\n\n"
                f"Conversation so far:\n{history_str}\n"
                f"Student just said: \"{student_text}\"\n\n"
                f"This is the FINAL message of the session. Acknowledge what the student "
                f"said and give a warm, brief closing remark that wraps up the chat about "
                f"{focus}. Recast any mistakes silently. Do NOT ask a question — the "
                f"conversation is ending. End on a calm, friendly closing note."
            )
        else:
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


# ── Sentence correction (per student turn) ────────────────────────────────────
def correct_sentence(student_text: str) -> str:
    """
    Return a fully corrected version of the student's sentence, or the sentence
    UNCHANGED if it is already natural and correct. Used for the review and the
    replay/practice feature. Conservative: only fixes genuine errors.
    """
    text = student_text.strip()
    if len(text.split()) < 2:
        return text  # too short to meaningfully correct

    system_prompt = """You are a careful English editor. You are given one sentence from
an English learner. Return the most natural, correct version of that sentence.

RULES:
- Fix only genuine errors: wrong verb tense, subject-verb disagreement, wrong or missing
  articles, wrong prepositions, and clearly unnatural phrasing.
- Keep the student's meaning and their words wherever possible — make the smallest changes
  needed to make it sound natural and correct.
- If the sentence is already natural and correct, return it EXACTLY as it is, unchanged.
- Do NOT change style or word choice just because you prefer a different word.
- Do NOT add or remove information. Do NOT make it longer or fancier.
- Optional contractions ("I am" / "I'm") and informal-but-correct expressions are fine —
  leave them alone.

Return ONLY the corrected sentence as plain text — no quotes, no labels, no explanation."""

    user_prompt = (
        f"Student sentence:\n{text}\n\n"
        f"Return the corrected sentence (or the same sentence if it is already correct)."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.2,
        )
        corrected = response.choices[0].message.content.strip()
        # Strip accidental wrapping quotes
        corrected = corrected.strip('"').strip()
        return corrected or text
    except Exception:
        return text


# ── Review builder ────────────────────────────────────────────────────────────
def build_review(turns: list, style: str = "clear", topic_name: str = "") -> str:
    """
    Build a review of the whole conversation as markdown.
    `turns` is a list of dicts: {morgan, student, corrected}.
    - morgan: Morgan's line (may be "" for the opening handled separately)
    - student: the student's original line
    - corrected: the corrected version (same as student if no change)
    Shows Morgan's lines, the student's original, and the correction where different.
    """
    if not turns:
        return "**Nice chat!** There's nothing to review yet."

    lines = ["## Conversation Review", ""]
    if topic_name:
        lines.append(f"*Topic: {topic_name}*")
        lines.append("")

    any_correction = False
    for t in turns:
        morgan_line    = (t.get("morgan") or "").strip()
        student_line   = (t.get("student") or "").strip()
        corrected_line = (t.get("corrected") or "").strip()

        if morgan_line:
            lines.append(f"**Morgan:** {morgan_line}")
        if student_line:
            lines.append(f"**You:** {student_line}")
            if corrected_line and corrected_line.lower() != student_line.lower():
                lines.append(f"**✓ Better:** {corrected_line}")
                any_correction = True
        lines.append("")

    lines.append("---")
    if any_correction:
        lines.append("The **✓ Better** lines show a more natural way to say what you said. "
                     "Try the practice round to say them out loud!")
    else:
        lines.append("Your English was natural throughout — wonderful work! "
                     "Try the practice round to say the conversation again.")
    return "\n".join(lines)
