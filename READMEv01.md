# 🗣️ English Conversation Coach

An AI-powered conversational English practice app that helps learners improve
naturally through real chat. Instead of explicit correction, the coach
silently models correct English through recasting — weaving the right forms
naturally into the conversation while a summary of errors is saved for review
at the end.

---

## How It Works

### During the conversation
The student chats freely with an AI coach — by voice or text — on everyday
topics. When the student makes a grammar error, uses unnatural phrasing, or
picks the wrong word, the coach naturally incorporates the correct version
into its own reply without drawing attention to it. This is called
**recasting** — a proven language teaching technique that exposes learners
to correct English without interrupting the flow of conversation.

### After the conversation
When the student clicks **End conversation**, the app generates a summary of
all errors caught during the chat. Each entry shows what the student said,
the correct version, and a brief explanation of the rule. The student reviews
the summary, then starts a new conversation.

---

## Two Coaching Styles

Students can switch between two coaches at any time using the toggle in the
navigation bar. The conversation resets when switching.

| Style | Coach | Voice | Best for |
|---|---|---|---|
| 💬 Casual | Dora | Canadian English | Intermediate to advanced learners who want exposure to natural native speech, idioms, and casual expressions |
| 📖 Clear | Morgan | US English | Beginner to intermediate learners who prefer clear, complete sentences and accessible vocabulary |

Both coaches use the same recasting technique — the difference is in how
naturally casual vs clear and structured their speech sounds.

---

## Features

- **Voice or text input** — speak or type your replies
- **Recasting** — errors corrected silently through natural conversation
- **Two coaching styles** — casual native English (Dora) or clear accessible English (Morgan)
- **Error summary** — full list of errors with corrections and explanations after each conversation
- **Style-aligned summary** — casual tone for Dora, structured tone for Morgan
- **No login required** — open and start chatting immediately
- **New conversation** button — resets the chat while preserving your chosen style

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| AI conversation + error analysis | Groq API (llama-3.1-8b-instant) |
| Speech-to-text | Groq Whisper turbo |
| Text-to-speech | gTTS (Canadian English + US English) |
| Hosting | Render.com |

---

## Project Structure

```
implicit-coach/
├── app.py                  # Flask routes and main application
├── implicit_agent.py       # Dora and Morgan agents, error capture, summary
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment configuration
├── .python-version         # Pins Python to 3.11
├── .gitignore
├── .env.example            # Example environment variables
└── templates/
    └── chat.html           # Chat UI with style toggle
```

---

## Getting Started

### Prerequisites
- Python 3.11
- A [Groq](https://console.groq.com) API key (free)

### Local Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/implicit-coach.git
   cd implicit-coach
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** based on `.env.example`
   ```
   GROQ_API_KEY=your_groq_api_key
   SECRET_KEY=any_random_string
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. Open `http://localhost:5000` in your browser

---

## Deploying to Render

1. Push the repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo
4. Set the following environment variables:
   - `GROQ_API_KEY` — your Groq API key
   - `SECRET_KEY` — any long random string
5. Set **Start Command** to:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
6. Deploy

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for LLM and Whisper transcription |
| `SECRET_KEY` | Flask session secret key |

---

## The Recasting Technique

Recasting is a well-established second language acquisition technique where
the teacher reformulates a learner's incorrect sentence correctly — without
explicitly pointing out the error. Research shows that learners acquire
correct forms more naturally when they hear them in context rather than being
told a rule.

**Example:**
- Student says: *"Yesterday I go to the market"*
- Coach says: *"Oh nice, when I went to the market last week I found the best
  mangoes — what did you pick up?"*

The student hears *"went"* used correctly in a natural sentence. The
conversation continues without interruption.

---

## License

MIT
