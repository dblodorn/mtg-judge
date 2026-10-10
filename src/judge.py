"""Judge listener — wires Moonshine STT → query router → hybrid RAG → LLM → TTS."""

import queue
import re
import threading

from moonshine_voice import TranscriptEventListener

from src.formats import display_name, parse_format, parse_switch

CONFIDENCE_PREFIX = {
    "high": "",
    "medium": "I'm fairly sure ",
    "low": "I'm not certain, but ",
}
FORMAT_PROMPT = "Which format are we playing? For example Modern, Legacy, Pioneer, or Commander."
MAX_FORMAT_TRIES = 2
# "Judge, switch to Legacy" alone is a command; anything longer also gets answered
SWITCH_ONLY_MAX_WORDS = 6


def strip_wake_word(text: str, wake_word: str | None) -> str | None:
    """Return the question after the wake word, or None if it wasn't addressed to us.

    "Judge, does trample..." → "does trample...". With wake_word=None every line
    is a question. A bare "Judge" returns "" (addressed, but no question yet).
    """
    if not wake_word:
        return text.strip()
    m = re.match(rf"^\s*(?:hey|okay|ok)?[\s,]*{re.escape(wake_word)}\b[\s,:.!?]*", text, re.IGNORECASE)
    if not m:
        return None
    return text[m.end():].strip()


class JudgeListener(TranscriptEventListener):
    """
    Moonshine transcript listener that:
    1. On start, asks which format is being played (unless one is configured);
       "Judge, switch to Legacy" changes it later
    2. Ignores anything not addressed to the wake word
    3. Hands work to a worker thread (keeps the mic callback fast)
    4. Worker: route → retrieve → LLM ruling → speak, with the mic muted
       while speaking so the judge doesn't transcribe itself
    """

    def __init__(self, retriever, router, llm, tts, mic=None, wake_word: str | None = "judge",
                 game_format: str | None = None):
        self.retriever = retriever
        self.router = router
        self.llm = llm
        self.tts = tts
        self.mic = mic
        self.wake_word = wake_word
        self.format = game_format
        # Work items: ("say", text) or ("ask", question). One pending item at most;
        # extra questions while busy are dropped.
        self._work: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
        self._awaiting_question = False
        self._format_tries = 0  # >0 while waiting for an answer to FORMAT_PROMPT
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Session

    def start(self):
        """Call once the mic is live: asks for the format if none is set."""
        if self.format is None:
            self._format_tries = 1
            self._queue("say", FORMAT_PROMPT)
        else:
            print(f"🃏 Format: {display_name(self.format)}")

    def set_format(self, fmt: str | None):
        self.format = fmt
        self._format_tries = 0
        print(f"🃏 Format: {display_name(fmt) if fmt else 'none'}")

    # ------------------------------------------------------------------
    # Transcript events (mic thread — keep fast)

    def on_line_text_changed(self, event):
        """Live transcript feedback — print on same line."""
        print(event.line.text, end="\r", flush=True)

    def on_line_completed(self, event):
        text = event.line.text
        print(f"\n👤 {text}")

        if self._format_tries:
            self._answer_format_prompt(text)
            return

        question = strip_wake_word(text, self.wake_word)
        if question is None and self._awaiting_question:
            # "Judge." ... "does trample..." — follow-up line after a bare wake word
            question = text.strip()
        if question is None:
            return
        if not question:
            self._awaiting_question = True
            self._queue("say", "Yes?")
            return
        self._awaiting_question = False

        switched = parse_switch(question)
        if switched:
            self.set_format(switched)
            if len(question.split()) <= SWITCH_ONLY_MAX_WORDS:
                self._queue("say", f"Got it, {display_name(switched)}.")
                return
        self._queue("ask", question)

    def _answer_format_prompt(self, text: str):
        """Any line answers the startup prompt — no wake word needed."""
        fmt = parse_format(text)
        if fmt:
            self.set_format(fmt)
            self._queue("say", f"Got it, {display_name(fmt)}. Say judge, then your question.")
        elif self._format_tries < MAX_FORMAT_TRIES:
            self._format_tries += 1
            self._queue("say", "Sorry, which format? " + FORMAT_PROMPT.split("? ", 1)[1])
        else:
            self.set_format(None)
            self._queue("say", "No problem. Say judge, switch to Modern, any time to set one.")

    def _queue(self, kind: str, payload: str):
        try:
            self._work.put_nowait((kind, payload))
        except queue.Full:
            print("⏳ Still working on the last question — ignoring this one.")

    # ------------------------------------------------------------------
    # Worker

    def _run(self):
        while True:
            kind, payload = self._work.get()
            if kind == "say":
                self._speak(payload)
            else:
                self.handle(payload)

    def handle(self, query: str) -> str | None:
        """Answer one question end to end. Returns the spoken answer."""
        try:
            route = self.router.route(query)
            route["format"] = self.format
            print(f"🔀 Route: {route['type']}")

            chunks = self.retriever.retrieve(query, route=route)
            print(f"📚 Retrieved {len(chunks)} chunks: {[c.get('id', '?') for c in chunks[:3]]}...")

            ruling = self.llm.answer(query, chunks, game_format=self.format)
            spoken_answer = CONFIDENCE_PREFIX.get(ruling.confidence, "") + ruling.answer
            print(f"⚖️  {spoken_answer}")
            if ruling.citations:
                print(f"   Citations: {', '.join(ruling.citations)}")

            self._speak(spoken_answer)
            return spoken_answer
        except Exception as e:
            print(f"⚠ Error processing query: {e}")
            self._speak("Sorry, I hit an error answering that.")
            return None

    def _speak(self, text: str):
        """Speak with the mic muted so the judge doesn't hear itself."""
        if self.tts is None:  # text-only mode (`--ask`)
            return
        if self.mic is not None:
            self.mic.mute(True)
        try:
            self.tts.say(text)
            self.tts.wait()
        finally:
            if self.mic is not None:
                self.mic.mute(False)
