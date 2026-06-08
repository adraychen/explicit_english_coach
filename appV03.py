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

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url, sslmode="require")

def get_next_topic():
    """Return the next topic for USER_NAME, sequentially by topic_order."""
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Find the highest topic_order this user has already completed
        cur.execute("""
            SELECT MAX(t.topic_order) AS done
            FROM eec_learning_log l
            JOIN eec_topics t ON l.topic_id = t.id
            WHERE l.user_name = %s
        """, (USER_NAME,))
        row  = cur.fetchone()
        done = row["done"] if row and row["done"] is not None else 0

        # Next topic in order; wrap around to the first if none higher
        cur.execute("""
            SELECT * FROM eec_topics WHERE topic_order > %s
            ORDER BY topic_order ASC LIMIT 1
        """, (done,))
        topic = cur.fetchone()
        if not topic:
            cur.execute("SELECT * FROM eec_topics ORDER BY topic_order ASC LIMIT 1")
            topic = cur.fetchone()
        return dict(topic) if topic else None
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
@app.route("/")
def index():
    style = request.args.get("style", session.get("style", "casual"))
    session["history"]       = []
    session["all_errors"]    = []
    session["last_question"] = ""
    session["style"]         = style
    session["taught_words"]  = []
    coach_name = "Dora" if style == "casual" else "Morgan"

    topic = None
    opening = "Say anything to start chatting!"
    if style == "clear":
        topic = get_next_topic()
        if topic:
            session["topic"]    = topic
            session["topic_id"] = topic["id"]
            opening = topic.get("opening") or opening

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

    opening = "Say anything to start chatting!"
    topic = None
    if style == "clear":
        topic = get_next_topic()
        if topic:
            session["topic"]    = topic
            session["topic_id"] = topic["id"]
            opening = topic.get("opening") or opening
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
        session_complete = len(taught_words) >= WORDS_PER_SESSION
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
        taught_words = session.get("taught_words", [])
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

        return jsonify({"summary": result})
    except Exception as e:
        print(f"[Summary] Error: {e}")
        return jsonify({
            "summary": "**Sorry, the summary could not be generated this time.** "
                       "Please try again, or start a new one."
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
