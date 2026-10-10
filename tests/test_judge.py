"""Tests for the judge listener (wake word, mic muting)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.judge import JudgeListener, acknowledgement, strip_wake_word
from src.llm import JudgeRuling


@pytest.mark.parametrize("text,expected", [
    ("Judge, does trample go through protection?", "does trample go through protection?"),
    ("hey judge: what is ward", "what is ward"),
    ("OK Judge. Can I respond?", "Can I respond?"),
    ("Judge.", ""),
    ("I'll attack with everything", None),
    ("don't judge me for this play", None),
    ("judgement day", None),
    ("A judge what happens", "what happens"),        # STT heard a filler word first
    ("The judge, what happens?", "what happens?"),
    ("Uh, judge. Is it legal?", "Is it legal?"),
])
def test_strip_wake_word(text, expected):
    assert strip_wake_word(text, "judge") == expected


def test_no_wake_word_answers_everything():
    assert strip_wake_word("  what is ward ", None) == "what is ward"


def _listener(wake_word="judge", game_format="modern", question_pause=0, thinking_cue="off", keepalive_seconds=12.0):
    llm = MagicMock()
    llm.answer.return_value = JudgeRuling(answer="Yes.", confidence="medium", citations=[])
    router = MagicMock()
    router.route.return_value = {"type": "general", "sources": None}
    listener = JudgeListener(
        retriever=MagicMock(), router=router, llm=llm, tts=MagicMock(), mic=MagicMock(), wake_word=wake_word,
        game_format=game_format, question_pause=question_pause,
        thinking_cue=thinking_cue, keepalive_seconds=keepalive_seconds,
    )
    listener._work = MagicMock()  # capture instead of handing to the worker
    return listener


def _queued(listener):
    return [c.args[0] for c in listener._work.put_nowait.call_args_list]


def _line(text):
    return SimpleNamespace(line=SimpleNamespace(text=text))


def test_ignores_lines_without_wake_word():
    listener = _listener()
    listener.on_line_completed(_line("pass the dice"))
    listener._work.put_nowait.assert_not_called()


def test_bare_wake_word_then_question():
    listener = _listener()
    listener.on_line_completed(_line("Judge?"))
    listener.on_line_completed(_line("does deathtouch work with trample"))

    assert _queued(listener) == [("chime", ""), ("ask", "does deathtouch work with trample")]


def test_handle_mutes_mic_while_speaking():
    listener = _listener()
    order = []
    listener.mic.mute.side_effect = lambda m=True: order.append(f"mute={m}")
    listener.tts.say.side_effect = lambda t: order.append("say")
    listener.tts.wait.side_effect = lambda: order.append("wait")

    spoken = listener.handle("does trample work?")

    assert spoken == "I'm fairly sure Yes."
    assert order == ["mute=True", "say", "wait", "mute=False"]


def test_startup_prompt_then_spoken_format():
    listener = _listener(game_format=None)
    listener.start()
    listener.on_line_completed(_line("We're playing EDH"))  # no wake word needed for the answer

    assert listener.format == "commander"
    assert _queued(listener)[0] == ("say", "Which format are we playing? For example Modern, Legacy, Pioneer, or Commander.")
    assert _queued(listener)[1][1].startswith("Got it, Commander.")


def test_unrecognized_format_reprompts_then_gives_up():
    listener = _listener(game_format=None)
    listener.start()
    listener.on_line_completed(_line("uh what"))
    listener.on_line_completed(_line("never mind"))

    assert listener.format is None
    assert listener._format_tries == 0  # back to normal wake-word mode
    assert [k for k, _ in _queued(listener)] == ["say", "say", "say"]


def test_switch_command_changes_format_without_asking():
    listener = _listener()
    listener.on_line_completed(_line("Judge, switch to Legacy"))

    assert listener.format == "legacy"
    assert _queued(listener) == [("say", "Got it, Legacy.")]


def test_switch_plus_question_sets_format_and_answers():
    listener = _listener()
    listener.on_line_completed(_line("Judge, we're playing Pioneer, is Thoughtseize legal in this format?"))

    assert listener.format == "pioneer"
    assert _queued(listener) == [("ask", "we're playing Pioneer, is Thoughtseize legal in this format?")]


def test_handle_passes_format_to_retriever_and_llm():
    listener = _listener(game_format="modern")
    listener.handle("is it legal?")

    route = listener.retriever.retrieve.call_args.kwargs["route"]
    assert route["format"] == "modern"
    assert listener.llm.answer.call_args.kwargs["game_format"] == "modern"


@pytest.mark.parametrize("text", ["Judge, stop", "Judge, stop listening.", "judge goodbye", "Judge, that's all",
                                  "Judge, shut down please", "The judge stopped."])
def test_stop_command(text):
    listener = _listener()
    listener.on_line_completed(_line(text))
    assert _queued(listener) == [("stop", "Goodbye.")]


@pytest.mark.parametrize("text", ["stop", "stop running.", "Judge, can I stop a trigger with Stifle?"])
def test_stop_needs_wake_word_and_whole_utterance(text):
    listener = _listener()
    listener.on_line_completed(_line(text))
    assert all(kind != "stop" for kind, _ in _queued(listener))


def test_worker_stop_speaks_goodbye_and_sets_stopped():
    listener = JudgeListener(retriever=MagicMock(), router=MagicMock(), llm=MagicMock(),
                             tts=MagicMock(), mic=MagicMock(), game_format="modern")
    listener._work.put(("stop", "Goodbye."))

    assert listener.stopped.wait(2)
    listener.tts.say.assert_called_with("Goodbye.")


def test_format_change_updates_speech_keyterms():
    listener = _listener(game_format=None)
    listener.start()
    prompt_terms = listener.mic.set_keyterms.call_args.args[0]
    listener.on_line_completed(_line("Modern"))
    modern_terms = listener.mic.set_keyterms.call_args.args[0]

    assert "Commander" in prompt_terms  # format names while asking
    assert "lore counter" in modern_terms and "Commander" not in modern_terms


def test_keyterm_failure_is_not_fatal():
    listener = _listener()
    listener.mic.set_keyterms.side_effect = RuntimeError("model is not streaming")
    listener.set_format("legacy")  # must not raise
    assert listener.format == "legacy"


def test_question_spanning_several_lines_is_joined():
    """'Judge, what happens when we counter the third,' / 'lore counter trigger' / 'on Urza's Saga'"""
    import time

    listener = _listener(question_pause=0.2)
    listener.on_line_completed(_line("Judge, what happens when we counter the third,"))
    listener.on_line_completed(_line("lore counter trigger"))
    listener.on_line_completed(_line("on Urza's Saga?"))
    assert _queued(listener) == []  # still listening

    time.sleep(0.4)
    assert _queued(listener) == [("ask", "what happens when we counter the third, lore counter trigger on Urza's Saga?")]


def test_lines_without_wake_word_are_ignored_when_not_composing():
    import time

    listener = _listener(question_pause=0.1)
    listener.on_line_completed(_line("lore counter trigger"))
    time.sleep(0.2)
    assert _queued(listener) == []


def test_stop_inside_a_question_does_not_quit():
    """'Judge, can I' … 'stop' (as part of a sentence) is part of the question."""
    import time

    listener = _listener(question_pause=0.2)
    listener.on_line_completed(_line("Judge, if my opponent says"))
    listener.on_line_completed(_line("stop"))
    time.sleep(0.4)
    assert _queued(listener) == [("ask", "if my opponent says stop")]


@pytest.mark.parametrize("text", ["Judge, stop.", "stop", "goodbye"])
def test_stop_during_format_prompt(text):
    listener = _listener(game_format=None)
    listener.start()
    listener.on_line_completed(_line(text))
    assert _queued(listener)[-1] == ("stop", "Goodbye.")


def test_spoken_acknowledgement_before_answer():
    listener = _listener(thinking_cue="speech")
    said = []
    listener.tts.say.side_effect = said.append

    listener.handle("does trample work?")

    assert said == ["Just a minute, looking up rulings for trample.", "I'm fairly sure Yes."]


def test_beep_acknowledgement():
    listener = _listener(thinking_cue="beep")
    listener.handle("does trample work?")

    listener.tts.play_success.assert_called_once()


def test_keepalive_while_llm_is_slow_then_stops():
    import time

    listener = _listener(thinking_cue="speech", keepalive_seconds=0.1)
    said = []
    listener.tts.say.side_effect = said.append
    original = listener.llm.answer.return_value
    listener.llm.answer.side_effect = lambda *a, **k: (time.sleep(0.35), original)[1]

    listener.handle("hard question")
    time.sleep(0.3)  # no cues after the answer

    assert said[0] == "Just a minute, looking up rulings for this scenario."
    assert said[1] == "Still thinking."
    assert said[-1] == "I'm fairly sure Yes."
    assert listener.tts.play_success.call_count >= 1  # chimes after the first reminder
    assert said.count("I'm fairly sure Yes.") == 1


def test_error_plays_error_tone():
    listener = _listener(thinking_cue="off")
    listener.retriever.retrieve.side_effect = RuntimeError("boom")

    assert listener.handle("q") is None
    listener.tts.play_error.assert_called_once()


@pytest.mark.parametrize("route,query,expected", [
    ({"mentioned_cards": ["Blood Moon", "Urza's Saga"]}, "what happens", "Blood Moon and Urza's Saga"),
    ({"rule_numbers": ["702.19c"]}, "what does it say", "rule 702.19c"),
    ({}, "how does deathtouch work with trample", "deathtouch and trample"),
    ({}, "can I respond to that", "this scenario"),
])
def test_acknowledgement_names_the_topic(route, query, expected):
    assert acknowledgement(query, route) == f"Just a minute, looking up rulings for {expected}."


@pytest.mark.parametrize("text", ["Judge, are you there?", "Judge, can you hear me?", "Judge, hello"])
def test_presence_check_gets_a_direct_reply(text):
    listener = _listener()
    listener.on_line_completed(_line(text))
    assert _queued(listener) == [("say", "I'm here. Ask me a rules question.")]


def test_words_still_arriving_keep_the_question_open():
    """Live bug: 'Saga out on the' was answered while '…battlefield, Blood Moon?' was still being spoken."""
    import time

    listener = _listener(question_pause=0.25)
    listener.on_line_completed(_line("Judge, I have Urza's Saga out on the"))
    for _ in range(4):  # 0.4 s of in-progress transcript updates, each under the pause
        time.sleep(0.1)
        listener.on_line_text_changed(_line("battlefield. What happens if my opponent plays"))
    assert _queued(listener) == []  # not answered mid-sentence

    listener.on_line_completed(_line("battlefield. What happens if my opponent plays Blood Moon?"))
    time.sleep(0.4)
    assert _queued(listener) == [("ask", "I have Urza's Saga out on the battlefield. What happens if my opponent plays Blood Moon?")]


def test_bare_wake_word_chimes_without_muting_mic():
    listener = JudgeListener(retriever=MagicMock(), router=MagicMock(), llm=MagicMock(),
                             tts=MagicMock(), mic=MagicMock(), game_format="modern", question_pause=0)
    listener.on_line_completed(_line("Judge."))
    import time
    time.sleep(0.2)

    listener.tts.play_success.assert_called_once()
    listener.mic.mute.assert_not_called()
