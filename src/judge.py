"""Judge listener — wires Moonshine STT → query router → hybrid RAG → LLM → TTS."""

from moonshine_voice import TranscriptEventListener


class JudgeListener(TranscriptEventListener):
    """
    Moonshine transcript listener that:
    1. Receives completed utterances from MicTranscriber
    2. Routes the query (card name, rule number, tournament, or general)
    3. Retrieves relevant MTG rules via hybrid RAG
    4. Gets a structured judge ruling from Ollama
    5. Speaks the answer via Moonshine TTS
    """

    def __init__(self, retriever, router, llm, tts, k: int = 8):
        self.retriever = retriever
        self.router = router
        self.llm = llm
        self.tts = tts
        self.k = k
        self.is_processing = False

    def on_line_text_changed(self, event):
        """Live transcript feedback — print on same line."""
        print(event.line.text, end="\r", flush=True)

    def on_line_completed(self, event):
        """User finished speaking — process the query."""
        query = event.line.text
        print(f"\n👤 {query}")

        if self.is_processing:
            return
        self.is_processing = True

        try:
            # Route the query
            route = self.router.route(query)
            print(f"🔀 Route: {route['type']}")

            # Retrieve relevant rules/cards
            chunks = self.retriever.retrieve(query, sources=route.get("sources"))
            print(f"📚 Retrieved {len(chunks)} chunks: {[c.get('id', '?') for c in chunks[:3]]}...")

            # Get structured ruling
            ruling = self.llm.answer(query, chunks)
            confidence_prefix = {
                "high": "",
                "medium": "I'm fairly sure ",
                "low": "I'm not certain, but ",
            }.get(ruling.confidence, "")
            spoken_answer = confidence_prefix + ruling.answer
            print(f"⚖️  {spoken_answer}")
            if ruling.citations:
                print(f"   Citations: {', '.join(ruling.citations)}")

            # Speak
            self.tts.say(spoken_answer)

        except Exception as e:
            print(f"⚠ Error processing query: {e}")
        finally:
            self.is_processing = False
