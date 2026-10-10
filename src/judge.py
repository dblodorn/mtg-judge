"""Judge listener — wires Moonshine STT → query router → hybrid RAG → LLM → TTS."""

import queue
import re
import threading

from moonshine_voice import TranscriptEventListener

from src.formats import display_name, parse_format, parse_switch
from src.keyterms import BASE_TERMS, keyterms_for

CONFIDENCE_PREFIX = {
    "high": "",
    "medium": "I'm fairly sure ",
    "low": "I'm not certain, but ",
}
# Spoken right after a question is understood, so the judge never seems to ignore you
STILL_THINKING = "Still thinking."
# "Judge, are you there?" — a check-in, not a rules question
PRESENCE_RE = re.compile(
    r"^(?:are you (?:there|listening|awake|thinking|on|working)|can you hear me|(?:hello|hi)(?: there)?|you there)[\s?.!]*$",
    re.IGNORECASE,
)
# Speech-to-text often hears a filler word before the wake word: "A judge, …", "The judge, …"
WAKE_FILLERS = r"(?:(?:hey|okay|ok|oh|uh|um|so|a|the)[\s,]+)*"
KEEPALIVE_SECONDS = 12.0  # cue again this often while the model is still reasoning
FORMAT_PROMPT = "Which format are we playing? For example Modern, Legacy, Pioneer, or Commander."
MAX_FORMAT_TRIES = 2
# "Judge, switch to Legacy" alone is a command; anything longer also gets answered
SWITCH_ONLY_MAX_WORDS = 6
# "Judge, stop" / "Judge, goodbye" — the whole utterance after the wake word
STOP_RE = re.compile(
    r"^(?:please\s+)?(?:stop(?:ped)?(?:\s+(?:listening|running))?|good\s?bye|shut\s?down|quit|exit|"
    r"that'?s all|we'?re done)(?:\s+please)?[\s.!]*$",
    re.IGNORECASE,
)


def _spoken_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def acknowledgement(query: str, route: dict) -> str:
    """'Just a minute, looking up rulings for Urza's Saga and Blood Moon.'

    Topic = the cards named, else the rule numbers, else game terms from the question
    (the same vocabulary speech recognition is biased toward), else "this scenario".
    """
    topics = list(route.get("mentioned_cards", []))[:3]
    if not topics:
        topics = [f"rule {n}" for n in route.get("rule_numbers", [])][:2]
    if not topics:
        lowered = query.lower()
        terms = [t for t in BASE_TERMS if re.search(rf"\b{re.escape(t.lower())}\b", lowered)]
        # Prefer longer terms ("lore counter" over "Saga"), keep question order
        terms = sorted(sorted(terms, key=len, reverse=True)[:2], key=lambda t: lowered.find(t.lower()))
        topics = terms
    subject = _spoken_list(topics) if topics else "this scenario"
    return f"Just a minute, looking up rulings for {subject}."


def strip_wake_word(text: str, wake_word: str | None) -> str | None:
    """Return the question after the wake word, or None if it wasn't addressed to us.

    "Judge, does trample..." → "does trample...". With wake_word=None every line
    is a question. A bare "Judge" returns "" (addressed, but no question yet).
    """
    if not wake_word:
        return text.strip()
    m = re.match(rf"^\s*{WAKE_FILLERS}{re.escape(wake_word)}\b[\s,:.!?]*", text, re.IGNORECASE)
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
                 game_format: str | None = None, question_pause: float = 3.0,
                 thinking_cue: str = "speech", keepalive_seconds: float = KEEPALIVE_SECONDS):
        self.retriever = retriever
        self.router = router
        self.llm = llm
        self.tts = tts
        self.mic = mic
        self.wake_word = wake_word
        self.format = game_format
        # Work items: ("say", text), ("chime", ""), ("ask", question) or ("stop", goodbye). One pending item at most;
        # extra questions while busy are dropped.
        self._work: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
        self._awaiting_question = False
        self._format_tries = 0  # >0 while waiting for an answer to FORMAT_PROMPT
        self.stopped = threading.Event()  # set after "Judge, stop" has been answered
        self._keyterms_warned = False
        # A question can span several transcript lines (people pause mid-sentence):
        # collect lines after the wake word until `question_pause` seconds of quiet.
        # 0 = answer each line immediately (tests).
        self.question_pause = question_pause
        self._pending: list[str] = []
        self._pending_lock = threading.Lock()
        self._pending_timer: threading.Timer | None = None
        # "speech" (short phrase), "beep" (success chime) or "off"
        self.thinking_cue = thinking_cue
        self.keepalive_seconds = keepalive_seconds
        self._speak_lock = threading.Lock()  # one voice at a time: cues never talk over answers
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Session

    def start(self):
        """Call once the mic is live: asks for the format if none is set."""
        self._update_keyterms(self.format)
        if self.format is None:
            self._format_tries = 1
            self._queue("say", FORMAT_PROMPT)
        else:
            print(f"🃏 Format: {display_name(self.format)}")

    def set_format(self, fmt: str | None):
        self.format = fmt
        self._format_tries = 0
        print(f"🃏 Format: {display_name(fmt) if fmt else 'none'}")
        self._update_keyterms(fmt)

    def _update_keyterms(self, fmt: str | None):
        """Bias speech recognition toward game words + this format's most-played cards."""
        if self.mic is None:
            return
        try:
            terms = keyterms_for(fmt)
            self.mic.set_keyterms(terms)
            print(f"🎯 Listening for {len(terms)} MTG terms")
        except Exception as e:  # e.g. a non-streaming STT model can't be biased
            if not self._keyterms_warned:
                print(f"⚠ Speech biasing unavailable ({e}); continuing without it.")
                self._keyterms_warned = True

    # ------------------------------------------------------------------
    # Transcript events (mic thread — keep fast)

    def on_line_text_changed(self, event):
        """Live transcript feedback — and the speaker is still talking, so keep waiting."""
        print(event.line.text, end="\r", flush=True)
        self._extend_pending()

    def on_line_started(self, event):
        self._extend_pending()

    def _extend_pending(self):
        """Restart the end-of-question timer while words are still arriving.

        Moonshine only completes a line at a pause, so a sentence can be mid-flight
        for a long time; timing out on completed lines alone answered half-questions.
        """
        with self._pending_lock:
            if self._pending_timer:
                self._pending_timer.cancel()
                self._start_pending_timer()

    def _start_pending_timer(self):
        # Caller holds _pending_lock
        self._pending_timer = threading.Timer(self.question_pause, self._flush_pending)
        self._pending_timer.daemon = True
        self._pending_timer.start()

    def on_line_completed(self, event):
        text = event.line.text
        print(f"\n👤 {text}")

        if self._format_tries:
            self._answer_format_prompt(text)
            return

        question = strip_wake_word(text, self.wake_word)
        with self._pending_lock:
            composing = bool(self._pending)
        if question is None and (self._awaiting_question or composing):
            # Continuation: the line after a bare "Judge", or more of a question in progress
            question = text.strip()
        if question is None:
            return
        if not question:
            # People keep talking right after "Judge" — a short chime with the mic left
            # open, not a spoken "Yes?" that mutes it and swallows their first words
            self._awaiting_question = True
            self._queue("chime", "")
            return
        self._awaiting_question = False

        if not composing and STOP_RE.match(question):
            self._queue("stop", "Goodbye.")
            return
        if not composing and PRESENCE_RE.match(question):
            self._queue("say", "I'm here. Ask me a rules question.")
            return

        if self.question_pause <= 0:
            self._dispatch(question)
            return
        with self._pending_lock:
            self._pending.append(question)
            if self._pending_timer:
                self._pending_timer.cancel()
            self._start_pending_timer()

    def _flush_pending(self):
        """Quiet for `question_pause` seconds: the question is complete."""
        with self._pending_lock:
            question = " ".join(self._pending).strip()
            self._pending = []
            self._pending_timer = None
        if question:
            self._dispatch(question)

    def _dispatch(self, question: str):
        """A complete question: a format switch, a question, or both."""
        switched = parse_switch(question)
        if switched:
            self.set_format(switched)
            if len(question.split()) <= SWITCH_ONLY_MAX_WORDS:
                self._queue("say", f"Got it, {display_name(switched)}.")
                return
        self._queue("ask", question)

    def _answer_format_prompt(self, text: str):
        """Any line answers the startup prompt — no wake word needed."""
        if STOP_RE.match(strip_wake_word(text, self.wake_word) or text.strip()):
            self._format_tries = 0
            self._queue("stop", "Goodbye.")
            return
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
            elif kind == "chime":
                if self.tts is not None:
                    self.tts.play_success()
            elif kind == "stop":
                self._speak(payload)
                self.stopped.set()
                return
            else:
                self.handle(payload)

    def handle(self, query: str) -> str | None:
        """Answer one question end to end. Returns the spoken answer."""
        done = threading.Event()
        try:
            route = self.router.route(query)
            route["format"] = self.format
            print(f"🔀 Route: {route['type']}")
            self._acknowledge(query, route)
            threading.Thread(target=self._keepalive, args=(done,), daemon=True).start()

            chunks = self.retriever.retrieve(query, route=route)
            print(f"📚 Retrieved {len(chunks)} chunks: {[c.get('id', '?') for c in chunks[:3]]}...")

            ruling = self.llm.answer(query, chunks, game_format=self.format)
            done.set()
            spoken_answer = CONFIDENCE_PREFIX.get(ruling.confidence, "") + ruling.answer
            print(f"⚖️  {spoken_answer}")
            if ruling.citations:
                print(f"   Citations: {', '.join(ruling.citations)}")

            self._speak(spoken_answer)
            return spoken_answer
        except Exception as e:
            done.set()  # stop cues before the apology
            print(f"⚠ Error processing query: {e}")
            self._speak("Sorry, I hit an error answering that.", beep="error")
            return None
        finally:
            done.set()

    # ------------------------------------------------------------------
    # Thinking cues

    def _acknowledge(self, query: str, route: dict):
        """Signal right away that the question was heard — and what it's about."""
        phrase = acknowledgement(query, route)
        print(f"🤔 {phrase}")
        if self.thinking_cue == "speech":
            self._speak(phrase)
        elif self.thinking_cue == "beep":
            self._speak(None, beep="success")

    def _keepalive(self, done: threading.Event):
        """While the model is still reasoning: 'Still thinking.' once, then a soft chime."""
        first = True
        while not done.wait(self.keepalive_seconds):
            if self.thinking_cue == "off":
                continue
            if first and self.thinking_cue == "speech":
                self._speak(STILL_THINKING, unless=done)
            else:
                self._speak(None, beep="success", unless=done)
            first = False

    # ------------------------------------------------------------------

    def _speak(self, text: str | None, beep: str | None = None, unless: threading.Event | None = None):
        """Speak (and/or play a "success"/"error" chime) with the mic muted so the
        judge doesn't hear itself. `unless`: skip if that event is set by the time
        it's our turn (a keep-alive cue that lost the race to the answer)."""
        if self.tts is None:  # text-only mode (`--ask`)
            return
        with self._speak_lock:
            if unless is not None and unless.is_set():
                return
            if self.mic is not None:
                self.mic.mute(True)
            try:
                if beep == "success":
                    self.tts.play_success()
                elif beep == "error":
                    self.tts.play_error()
                if text:
                    self.tts.say(text)
                self.tts.wait()
            finally:
                if self.mic is not None:
                    self.mic.mute(False)
