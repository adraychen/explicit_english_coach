# 🗣️ English Conversation Coach

An AI-powered conversational English practice app that helps learners build
fluency by speaking with an engaging AI coach. The app combines two
approaches: free casual conversation with silent error correction, and
topic-led practice where the coach models useful everyday phrases in a clear,
podcast-style conversation.

---

## The Two Coaches

Students switch between two coaches at any time using the toggle in the
navigation bar. Switching resets the conversation.

| Coach | Style | Voice | What it's for |
|---|---|---|---|
| 💬 Dora | Casual | Canadian English | Free, natural conversation on any subject. Dora chats like a friend, recasts errors silently, and uses rich everyday vocabulary. Good for learners who want exposure to natural native speech. |
| 📖 Morgan | Clear, topic-led | US English | Structured practice on a set topic. Morgan hosts a clear, engaging conversation — like an English-learning podcast host — modelling useful words and phrases at the topic's difficulty level. Good for learners building everyday vocabulary and confidence. |

Both coaches use **recasting** — silently weaving the correct form into the
reply instead of pointing out mistakes — and both produce an error review at
the end of the session.

---

## How It Works

### Dora — free conversation
The student chats freely with Dora by voice or text. Dora keeps the
conversation natural and engaging, silently recasting any errors. The student
ends the conversation whenever they like and sees a review of the errors
caught during the chat.

### Morgan — topic-led practice
Morgan leads a clear, engaging conversation on a topic drawn from the
database. Each topic has a vocabulary pool of useful words and phrases, a
difficulty level, and a focus keyword that keeps the conversation on subject.
Morgan:

- Speaks like a warm, clever podcast host (think Leo & Tina) — not a classroom teacher
- Models the topic's words and phrases naturally so the student hears and absorbs them
- Adapts language complexity to the topic's difficulty level (e.g. simple sentences for A1 Beginner)
- Acknowledges what the student said and keeps the conversation flowing on topic
- Recasts errors silently as they come up

A Morgan session runs for a set number of exchanges, then ends with a review.

### The review
At the end of a session the app shows a review:

- **For Morgan** — the words and phrases practised this session (with simple explanations and examples) plus any errors to work on
- **For Dora** — a friendly summary of the errors caught during the chat

The words Morgan used are recorded so future sessions can introduce new
vocabulary rather than repeating what's already been covered.

---

## Topics & Progression (Morgan)

Topics live in the database and are taught in sequence. A topic has a pool of
vocabulary, so one topic can span several sessions — each session introduces
fresh words. When a topic's vocabulary has been covered, Morgan moves on to
the next topic. Learning is tracked per user so progress carries across
sessions.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| Database | PostgreSQL (Supabase) — topics and learning log |
| Dora conversation + error capture | Groq API (llama-3.1-8b-instant) |
| Morgan conversation + review | Groq API (llama-3.3-70b-versatile) |
| Speech-to-text | Groq Whisper turbo |
| Text-to-speech | gTTS (Canadian English for Dora, US English for Morgan) |
| Hosting | Render.com |

---

## Database Tables

| Table | Description |
|---|---|
| `eec_topics` | Topics for Morgan: order, name, level, opening line, vocabulary pool, focus keyword, and sample coach views |
| `eec_learning_log` | Records which words each user has practised, per topic, for cross-session progression |

---

## Project Structure

```
implicit-coach/
├── app.py                    # Flask routes, DB access, session logic
├── implicit_agent.py         # Dora & Morgan prompts, error capture, word tracking, review
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment configuration
├── .python-version           # Pins Python to 3.11
├── .gitignore
├── .env.example              # Example environment variables
├── setup_tables.sql          # Creates eec_topics and eec_learning_log + first topic
├── add_focus_keyword.sql     # Adds the focus_keyword column to eec_topics
└── templates/
    └── chat.html             # Chat UI with coach toggle and review modal
```

---

## Getting Started

### Prerequisites
- Python 3.11
- A [Groq](https://console.groq.com) API key (free)
- A [Supabase](https://supabase.com) project (free)

### Database setup
Run `setup_tables.sql` in the Supabase SQL editor to create the tables and the
first topic, then run `add_focus_keyword.sql` to add the focus keyword column.

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
   DATABASE_URL=your_supabase_connection_string
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
4. Set these environment variables:
   - `GROQ_API_KEY` — your Groq API key
   - `SECRET_KEY` — any long random string
   - `DATABASE_URL` — your Supabase connection string (transaction pooler, port 6543)
5. Set **Start Command** to:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
6. Deploy

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for the LLMs and Whisper transcription |
| `SECRET_KEY` | Flask session secret key |
| `DATABASE_URL` | Supabase PostgreSQL connection string (transaction pooler) |

---

## The Recasting Technique

Recasting is a well-established second-language teaching technique where the
coach reformulates a learner's incorrect sentence correctly — without
explicitly pointing out the error. Learners acquire correct forms more
naturally when they hear them in context rather than being told a rule.

**Example:**
- Student says: *"Yesterday I go to the market"*
- Coach says: *"Oh nice, you went to the market yesterday? What did you pick up?"*

The student hears *"went"* used correctly, and the conversation continues
without interruption.

---

## Roadmap

- Shadowing practice for sentence patterns
- User login (currently single-user; learning is tracked under one name)
- More topics across more difficulty levels

---

## License

MIT
