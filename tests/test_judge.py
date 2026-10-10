"""Tests for the judge listener (wake word, mic muting)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.judge import JudgeListener, strip_wake_word
from src.llm import JudgeRuling


@pytest.mark.parametrize("text,expected", [
    ("Judge, does trample go through protection?", "does trample go through protection?"),
    ("hey judge: what is ward", "what is ward"),
    ("OK Judge. Can I respond?", "Can I respond?"),
    ("Judge.", ""),
    ("I'll attack with everything", None),
    ("don't judge me for this play", None),
    ("judgement day", None),
])
def test_strip_wake_word(text, expected):
    assert strip_wake_word(text, "judge") == expected


def test_no_wake_word_answers_everything():
    assert strip_wake_word("  what is ward ", None) == "what is ward"


def _listener(wake_word="judge", game_format="modern"):
    llm = MagicMock()
    llm.answer.return_value = JudgeRuling(answer="Yes.", confidence="medium", citations=[])
    router = MagicMock()
    router.route.return_value = {"type": "general", "sources": None}
    listener = JudgeListener(
        retriever=MagicMock(), router=router, llm=llm, tts=MagicMock(), mic=MagicMock(), wake_word=wake_word,
        game_format=game_format,
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

    assert _queued(listener) == [("say", "Yes?"), ("ask", "does deathtouch work with trample")]


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
