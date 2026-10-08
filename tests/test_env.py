import os

from insider_screen import load_env


def test_load_env(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nA_KEY=abc\nexport B_KEY="x y"\nC_KEY=from_file\n\nnot a line\n', encoding="utf-8")
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.setenv("C_KEY", "from_shell")
    load_env(f)
    assert os.environ["A_KEY"] == "abc" and os.environ["B_KEY"] == "x y"
    assert os.environ["C_KEY"] == "from_shell"
    for k in ("A_KEY", "B_KEY"):
        monkeypatch.delenv(k)
