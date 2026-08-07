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


def test_index_then_search_from_saved_index(tmp_path, capsys):
    index_path = str(tmp_path / "index.json")
    assert main(["--corpus", CORPUS, "index", "--out", index_path]) == 0
    assert os.path.exists(index_path)
    capsys.readouterr()
    assert (
        main(
            [
                "--corpus",
                CORPUS,
                "--index",
                index_path,
                "search",
                "reciprocal rank fusion",
                "-k",
                "2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "rank-fusion" in out


def test_saved_and_fresh_searches_agree(tmp_path, capsys):
    index_path = str(tmp_path / "index.json")
    main(["--corpus", CORPUS, "index", "--out", index_path])
    capsys.readouterr()
    main(["--corpus", CORPUS, "search", "sentence overlap in chunking", "-k", "3"])
    fresh = capsys.readouterr().out
    main(["--corpus", CORPUS, "--index", index_path, "search", "sentence overlap in chunking", "-k", "3"])
    loaded = capsys.readouterr().out
    assert loaded == fresh


def test_stale_index_fails_loudly(tmp_path, capsys):
    # Build an index from a modified copy of the corpus, then query it
    # against the real corpus directory: the fingerprints must not match.
    import shutil

    corpus_copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus_copy)
    with open(corpus_copy / "rag.md", "a", encoding="utf-8") as handle:
        handle.write("\nAn extra sentence the real corpus does not have.\n")
    index_path = str(tmp_path / "index.json")
    assert main(["--corpus", str(corpus_copy), "index", "--out", index_path]) == 0
    capsys.readouterr()
    from ragkb.store import StaleIndexError

    with pytest.raises(StaleIndexError):
        main(["--corpus", CORPUS, "--index", index_path, "search", "anything"])
