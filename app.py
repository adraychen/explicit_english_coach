import os
import io
import base64
import tempfile
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from groq import Groq
from gtts import gTTS
from implicit_agent import get_chat_response, capture_errors, generate_summary

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── Audio helpers ─────────────────────────────────────────────────────────────
def make_audio_b64(text: str, lang: str = "en") -> str:
    if not text:
        print("[Audio] No text provided")
        return ""
    text = text[:500]
    for attempt in range(3):
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            audio_data = base64.b64encode(buf.read()).decode()
            print(f"[Audio] Generated {len(audio_data)} bytes for: {text[:50]}...")
            return audio_data
        except Exception as e:
            print(f"[Audio] Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
    print("[Audio] All attempts failed")
    return ""

def jamie_audio(text: str) -> str:
    return make_audio_b64(text, lang="en-gb")

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
    session["history"]     = []
    session["all_errors"]  = []
    session["last_question"] = ""
    return render_template("chat.html")

@app.route("/respond", methods=["POST"])
def respond():
    data         = request.get_json()
    student_text = data.get("text", "").strip()
    if not student_text:
        return jsonify({"error": "No text provided"}), 400

    history      = session.get("history", [])
    last_question = session.get("last_question", "")
    all_errors   = session.get("all_errors", [])

    # Get Jamie's reply and capture errors
    reply  = get_chat_response(student_text, history)
    errors = capture_errors(student_text, last_question)

    # Accumulate errors
    all_errors.extend(errors)

    # Update history
    history.append({"role": "student",   "content": student_text})
    history.append({"role": "assistant", "content": reply})

    session["history"]      = history
    session["all_errors"]   = all_errors
    session["last_question"] = reply

    return jsonify({
        "reply":       reply,
        "reply_audio": jamie_audio(reply),
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
    all_errors = session.get("all_errors", [])
    result     = generate_summary(all_errors)
    return jsonify({"summary": result})

@app.route("/new")
def new_conversation():
    session.clear()
    return jsonify({"redirect": "/"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
