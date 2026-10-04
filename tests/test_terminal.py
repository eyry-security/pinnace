"""Terminal narration rendering tests — no real TTY required."""

from __future__ import annotations

import io

from pinnace.terminal import TerminalRenderer


class FakeStream(io.StringIO):
    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self.tty = tty
        self.flushes = 0

    def isatty(self) -> bool:
        return self.tty

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def test_redirected_output_has_progress_markers_without_ansi():
    stream = FakeStream(tty=False)
    renderer = TerminalRenderer(stream)

    assert renderer.render("[pinnace] turn 2/9, calling model…") == (
        "◆ turn 2/9, calling model…"
    )
    assert "\033[" not in renderer.render("[pinnace] tool: shell({'command': 'pwd'})")
    assert renderer.render("[pinnace] compacted context (100→20 est. tokens)").startswith("↻ ")
    assert renderer.render("[pinnace] finish() called — done").startswith("✓ ")
    assert renderer.render("error: model unavailable").startswith("✗ ")


def test_tty_colors_marker_but_preserves_message(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    renderer = TerminalRenderer(FakeStream(tty=True))

    rendered = renderer.render("[pinnace] tool: shell({'command': 'pwd'})")
    assert rendered.startswith("\033[33m→\033[0m ")
    assert rendered.endswith("tool: shell({'command': 'pwd'})")


def test_no_color_environment_disables_ansi(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    rendered = TerminalRenderer(FakeStream(tty=True)).render(
        "[pinnace] no tool calls — done"
    )
    assert rendered.startswith("✓ ")
    assert "\033[" not in rendered


def test_multiline_messages_are_indented_and_call_flushes():
    stream = FakeStream(tty=False)
    renderer = TerminalRenderer(stream)

    renderer("error: first line\nsecond line")

    assert stream.getvalue() == "✗ error: first line\n  second line\n"
    assert stream.flushes == 1


def test_session_and_unknown_events_remain_scannable():
    renderer = TerminalRenderer(FakeStream(tty=False))
    assert renderer.render("[pinnace] resumed session 'demo' (4 messages)").startswith("● ")
    assert renderer.render("plain informational message") == "• plain informational message"
