"""CLI entry point coverage for interactive scenario selection."""

from recovera.cli import _interactive_scenario


def test_interactive_menu_dispatches_all_five_scenarios(monkeypatch, capsys):
    expected = {
        "1": "baseline",
        "2": "drop-demo",
        "3": "obstacle-replan",
        "4": "blocked-route",
        "5": "beyond-reach",
    }
    for selection, scenario in expected.items():
        monkeypatch.setattr("builtins.input", lambda _prompt, value=selection: value)
        assert _interactive_scenario() == scenario
    output = capsys.readouterr().out
    assert "RECOVERA SIMULATION" in output
    assert "Obstacle Avoidance" in output


def test_interactive_menu_exit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "0")
    assert _interactive_scenario() is None
