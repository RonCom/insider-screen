import pytest


@pytest.fixture(autouse=True)
def _no_ollama_env(monkeypatch):
    """OLLAMA_THINK in the shell (set for an extraction run) must not change what the tests check."""
    from insider_screen import extract
    monkeypatch.setattr(extract, "THINK", None)
    monkeypatch.setattr(extract, "THINK_FALLBACK", False)
