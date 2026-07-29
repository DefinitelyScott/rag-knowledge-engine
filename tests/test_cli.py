import os

import pytest

from ragkb.cli import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")


def test_stats_command(capsys):
    assert main(["--corpus", CORPUS, "stats"]) == 0
    out = capsys.readouterr().out
    assert "documents" in out and "chunks" in out


def test_search_command(capsys):
    assert main(["--corpus", CORPUS, "search", "reciprocal rank fusion", "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "rank-fusion" in out


def test_ask_command(capsys):
    assert main(["--corpus", CORPUS, "ask", "what does the b parameter control"]) == 0
    out = capsys.readouterr().out
    assert "Sources:" in out


def test_missing_subcommand_exits():
    with pytest.raises(SystemExit):
        main(["--corpus", CORPUS])
