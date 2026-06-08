import os
import io
import base64
import tempfile
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from groq import Groq
from gtts import gTTS
import psycopg2
import psycopg2.extras
from implicit_agent import (
    get_chat_response, capture_errors, generate_summary, track_teaching
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

USER_NAME       = "Ray"   # Option C: single user for now; becomes user_id with login
WORDS_PER_SESSION = 5     # Morgan ends the session after teaching this many
MAX_MORGAN_TURNS  = 8     # Hard safety cap: never run a session longer than this

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url, sslmode="require")

def _single_words(vocabulary_pool: str) -> list:
    """Return only the single vocabulary words from a pool (excludes [..] patterns)."""
    return [w.strip() for w in (vocabulary_pool or "").split("\n")
            if w.strip() and "[" not in w]

def get_next_topic():
    """
    Return the next topic for USER_NAME plus the words already taught for it.
    Advances through topics in order: a topic is 'complete' when all its single
    vocabulary words have been taught. Returns dict with an extra key
    'already_taught' (list of words taught for this topic so far).
    """
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM eec_topics ORDER BY topic_order ASC")
        topics = cur.fetchall()
        if not topics:
            return None

        for topic in topics:
            topic = dict(topic)
            words = _single_words(topic.get("vocabulary_pool", ""))

            # Which of this topic's words has the user already been taught?
            cur.execute("""
                SELECT DISTINCT word_taught FROM eec_learning_log
                WHERE user_name = %s AND topic_id = %s
            """, (USER_NAME, topic["id"]))
            taught = {r["word_taught"] for r in cur.fetchall()}
            taught_single = [w for w in words if w in taught]

            # If not all single words are taught, this is the active topic
            if len(taught_single) < len(words):
                topic["already_taught"] = taught_single
                return topic

        # All topics fully taught — cycle back to the first, fresh
        first = dict(topics[0])
        first["already_taught"] = []
        return first
    finally:
        conn.close()

def log_learning(topic_id, taught_words, errors_made):
    """Write taught words to eec_learning_log for USER_NAME."""
    if not taught_words:
        return
    conn = get_db()
    try:
        cur = conn.cursor()
        had_error = bool(errors_made)
        for word in taught_words:
            cur.execute("""
                INSERT INTO eec_learning_log (user_name, topic_id, word_taught, had_error)
                VALUES (%s, %s, %s, %s)
            """, (USER_NAME, topic_id, word, had_error))
        conn.commit()
    finally:
        conn.close()

# ── Audio helpers ─────────────────────────────────────────────────────────────
def make_audio_b64(text: str, lang: str = "en") -> str:
    if not text:
        return ""
    text = text[:500]
    for attempt in range(3):
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception as e:
            print(f"[Audio] Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
    return ""

def coach_audio(text: str, style: str = "casual") -> str:
    lang = "en-ca" if style == "casual" else "en"
    return make_audio_b64(text, lang=lang)

def build_spoken_summary(taught_words: list, all_errors: list) -> str:
    """A short, natural spoken recap for Morgan to read aloud (kept within TTS limits)."""
    parts = ["Great work today! Let's quickly go over what we practised."]

    # Only speak the actual vocabulary words, not the sentence patterns
    words = [w for w in (taught_words or []) if "[" not in w]
    if words:
        if len(words) == 1:
            parts.append(f"You learned the word {words[0]}.")
        else:
            parts.append("You learned these words: " + ", ".join(words[:-1]) + f", and {words[-1]}.")
        # Reinforce by repeating them slowly once more
        parts.append("Let's say them again: " + ". ".join(words) + ".")

    if all_errors:
        parts.append("I also have a few small tips for you to look at on the screen.")
    else:
        parts.append("You made no big mistakes today, which is wonderful.")

    parts.append("Keep practising, and read the full review on your screen. See you next time!")
    spoken = " ".join(parts)
    return spoken[:480]  # keep within gTTS per-request limit

def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        with open(f.name, "rb") as af:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=af,
                language="en",
            )
    return result.text.strip()

# ── Routes ────────────────────────────────────────────────────────────────────
def setup_morgan_session():
    """
    Load the active topic, seed already-taught words, and choose an opening line.
    Returns (topic_dict_or_None, opening_text).
    """
    topic = get_next_topic()
    if not topic:
        return None, "Say anything to start chatting!"

    already = topic.get("already_taught", [])
    session["topic"]        = topic
    session["topic_id"]     = topic["id"]
    session["taught_words"] = list(already)  # exclude these from teaching

    if already:
        # Continuing session — generate a natural "welcome back" opening
        remaining = [w for w in _single_words(topic.get("vocabulary_pool", ""))
                     if w not in already]
        opening = (
            f"Welcome back! Last time we talked about feelings like "
            f"{', '.join(already[:3])}. Today let's keep going with a few more. "
            f"To start — how are you feeling right now?"
        )
        if topic.get("name", "").lower() != "talking about feelings":
            # Generic continuing opening for other topics
            opening = (
                f"Welcome back! Let's continue with {topic.get('name','our topic')}. "
                f"How are you doing today?"
            )
    else:
        # First session for this topic — use the stored opening
        opening = topic.get("opening") or "Say anything to start chatting!"

    return topic, opening

@app.route("/")
def index():
    style = request.args.get("style", session.get("style", "casual"))
    session["history"]       = []
    session["all_errors"]    = []
    session["last_question"] = ""
    session["style"]         = style
    session["taught_words"]  = []
    session["morgan_turns"]  = 0
    session["taught_this_session"] = []
    coach_name = "Dora" if style == "casual" else "Morgan"

    opening = "Say anything to start chatting!"
    if style == "clear":
        _, opening = setup_morgan_session()

    return render_template("chat.html", style=style, coach_name=coach_name,
                           opening=opening)

@app.route("/set_style", methods=["POST"])
def set_style():
    data  = request.get_json()
    style = data.get("style", "casual")
    session["history"]       = []
    session["all_errors"]    = []
    session["last_question"] = ""
    session["style"]         = style
    session["taught_words"]  = []
    session["morgan_turns"]  = 0
    session["taught_this_session"] = []

    opening = "Say anything to start chatting!"
    if style == "clear":
        _, opening = setup_morgan_session()
    else:
        session.pop("topic", None)
        session.pop("topic_id", None)

    return jsonify({
        "style":      style,
        "coach_name": "Dora" if style == "casual" else "Morgan",
        "opening":    opening,
        "opening_audio": coach_audio(opening, style) if style == "clear" else "",
    })

@app.route("/respond", methods=["POST"])
def respond():
    data         = request.get_json()
    student_text = data.get("text", "").strip()
    if not student_text:
        return jsonify({"error": "No text provided"}), 400

    history       = session.get("history", [])
    last_question = session.get("last_question", "")
    all_errors    = session.get("all_errors", [])
    style         = session.get("style", "casual")
    taught_words  = session.get("taught_words", [])
    topic         = session.get("topic")

    # Always capture errors (both coaches)
    errors = capture_errors(student_text, last_question)
    all_errors.extend(errors)

    if style == "clear" and topic:
        # Morgan — topic-led teaching
        reply = get_chat_response(student_text, history, style,
                                  topic=topic, taught_words=taught_words)
        # Track which target words Morgan just taught
        newly_taught = track_teaching(reply, topic.get("vocabulary_pool", ""), taught_words)
        taught_words = taught_words + newly_taught
        session["taught_words"] = taught_words

        # Track how many words were taught THIS session (excludes the seeded
        # words already learned in previous sessions).
        taught_this_session = session.get("taught_this_session", [])
        taught_this_session = taught_this_session + newly_taught
        session["taught_this_session"] = taught_this_session

        # Count Morgan turns as a safety net
        morgan_turns = session.get("morgan_turns", 0) + 1
        session["morgan_turns"] = morgan_turns

        # End the session once enough NEW words are taught — but not in the middle
        # of a question. If Morgan's reply ends with a question, let the student
        # answer first; the session will complete on the next turn.
        reached_target     = len(taught_this_session) >= WORDS_PER_SESSION
        ends_with_question = reply.rstrip().endswith("?")
        session_complete   = reached_target and not ends_with_question

        # Hard safety cap — never let a session run longer than this many turns,
        # even if word tracking fails for some reason.
        if morgan_turns >= MAX_MORGAN_TURNS:
            session_complete = True

        session["reached_target"] = reached_target
    else:
        # Dora — free chat
        reply = get_chat_response(student_text, history, style)
        session_complete = False

    history.append({"role": "student",   "content": student_text})
    history.append({"role": "assistant", "content": reply})

    session["history"]       = history
    session["all_errors"]    = all_errors
    session["last_question"] = reply

    return jsonify({
        "reply":            reply,
        "reply_audio":      coach_audio(reply, style),
        "session_complete": session_complete,
        "taught_count":     len(taught_words) if style == "clear" else 0,
    })

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    audio_data = request.data
    if not audio_data:
        return jsonify({"error": "No audio"}), 400
    try:
        text = transcribe(audio_data)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summary", methods=["POST"])
def summary():
    try:
        all_errors   = session.get("all_errors", [])
        style        = session.get("style", "casual")
        # Only the words taught THIS session (not previously learned ones)
        taught_words = session.get("taught_this_session", [])
        topic        = session.get("topic") or {}
        topic_name   = topic.get("name", "")
        topic_id     = session.get("topic_id")

        result = generate_summary(all_errors, style,
                                  taught_words=taught_words, topic_name=topic_name)

        # Log Morgan's taught words to the database
        if style == "clear" and taught_words and topic_id:
            try:
                log_learning(topic_id, taught_words, all_errors)
            except Exception as e:
                print(f"[Learning log] Error: {e}")

        # Morgan speaks the review aloud (clear style only).
        # gTTS has a length limit, so speak a short spoken intro + the taught words.
        summary_audio = ""
        if style == "clear":
            spoken = build_spoken_summary(taught_words, all_errors)
            summary_audio = coach_audio(spoken, style)

        return jsonify({"summary": result, "summary_audio": summary_audio})
    except Exception as e:
        print(f"[Summary] Error: {e}")
        return jsonify({
            "summary": "**Sorry, the summary could not be generated this time.** "
                       "Please try again, or start a new one.",
            "summary_audio": ""
        }), 200

@app.route("/new")
def new_conversation():
    style = session.get("style", "casual")
    session.clear()
    session["style"] = style
    return jsonify({"redirect": "/"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
