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
    get_chat_response, correct_sentence, build_review, words_used_in_text
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

USER_NAME       = "Ray"   # Option C: single user for now; becomes user_id with login
MAX_EXCHANGES   = 6       # Morgan session ends after this many exchanges, then review

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
    session["last_question"] = ""
    session["turns"]         = []
    session["style"]         = style
    session["taught_words"]  = []
    session["exchanges"]     = 0
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
    session["last_question"] = ""
    session["turns"]         = []
    session["style"]         = style
    session["taught_words"]  = []
    session["exchanges"]     = 0

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
    style         = session.get("style", "casual")
    taught_words  = session.get("taught_words", [])
    topic         = session.get("topic")
    turns         = session.get("turns", [])

    if style == "clear" and topic:
        # Decide whether THIS reply should be the closing one.
        exchanges = session.get("exchanges", 0) + 1
        session["exchanges"] = exchanges
        is_closing = exchanges >= MAX_EXCHANGES

        # Morgan reply (closing turn has no question)
        reply = get_chat_response(student_text, history, style,
                                  topic=topic, taught_words=taught_words,
                                  is_closing=is_closing)

        # Correct the student's sentence (full-sentence, conservative)
        corrected = correct_sentence(student_text)

        # Store a turn record for the review and the replay/practice feature
        turns.append({
            "morgan":    session.get("last_question", ""),  # Morgan's line they replied to
            "student":   student_text,
            "corrected": corrected,
            "reply":     reply,                              # Morgan's new line
        })
        session["turns"] = turns

        session_complete = is_closing
    else:
        # Dora — free chat (no review/practice, no corrections stored)
        reply = get_chat_response(student_text, history, style)
        session_complete = False

    history.append({"role": "student",   "content": student_text})
    history.append({"role": "assistant", "content": reply})

    session["history"]       = history
    session["last_question"] = reply

    return jsonify({
        "reply":            reply,
        "reply_audio":      coach_audio(reply, style),
        "session_complete": session_complete,
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
        style        = session.get("style", "casual")
        history      = session.get("history", [])
        turns        = session.get("turns", [])
        topic        = session.get("topic") or {}
        topic_name   = topic.get("name", "")
        topic_id     = session.get("topic_id")
        pool         = topic.get("vocabulary_pool", "")

        # Build the full-conversation review from the stored turn records
        result = build_review(turns, style, topic_name)

        # Silently log which vocabulary words Morgan used, for topic progression.
        if style == "clear" and pool and topic_id:
            morgan_text = " ".join(
                m["content"] for m in history if m.get("role") == "assistant"
            )
            taught_words = words_used_in_text(morgan_text, pool)
            if taught_words:
                try:
                    log_learning(topic_id, taught_words, [])
                except Exception as e:
                    print(f"[Learning log] Error: {e}")

        # Expose the corrected conversation for the replay/practice feature.
        # Temporary (session only) — not saved to the database.
        practice_turns = [
            {
                "morgan":    t.get("morgan", ""),
                "student":   t.get("student", ""),
                "corrected": t.get("corrected", ""),
                "reply":     t.get("reply", ""),
            }
            for t in turns
        ]

        return jsonify({"summary": result, "practice_turns": practice_turns})
    except Exception as e:
        print(f"[Summary] Error: {e}")
        return jsonify({
            "summary": "**Sorry, the review could not be generated this time.** "
                       "Please try again, or start a new one.",
            "practice_turns": []
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
