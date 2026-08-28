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


def test_ngram_flag_reaches_the_subcommand(capsys):
    """Regression guard on an argparse footgun that shipped once.

    ``--ngram`` was first declared with ``nargs="?"``, which makes argparse
    consume the *subcommand* as its value: ``--ngram search ...`` failed with
    "invalid int value: 'search'". A flag that only works when it is the last
    argument is a broken flag, so the ordering is asserted rather than assumed.
    """
    assert main(["--corpus", CORPUS, "--ngram", "4", "stats"]) == 0
    assert "ngram_vocabulary" in capsys.readouterr().out
    assert (
        main(["--corpus", CORPUS, "--ngram", "4", "search", "rank fusion", "-k", "2"])
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "--corpus",
                CORPUS,
                "--ngram",
                "4",
                "search",
                "rank fusion",
                "-k",
                "2",
                "--method",
                "ngram",
            ]
        )
        == 0
    )


def test_saved_index_cannot_gain_an_ngram_list_by_flag(tmp_path, capsys):
    index_path = str(tmp_path / "index.json")
    assert main(["--corpus", CORPUS, "index", "--out", index_path]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["--corpus", CORPUS, "--ngram", "4", "--index", index_path, "stats"])


def test_serve_command_parses_with_defaults():
    """The blocking loop is not run here; only that the flags reach it."""
    from ragkb.cli import build_parser
    from ragkb.serve import DEFAULT_HOST, DEFAULT_PORT

    args = build_parser().parse_args(["--corpus", CORPUS, "serve"])
    assert args.host == DEFAULT_HOST
    assert args.port == DEFAULT_PORT
    assert args.log_requests is False


def test_serve_command_accepts_host_and_port():
    from ragkb.cli import build_parser

    args = build_parser().parse_args(
        ["serve", "--host", "0.0.0.0", "--port", "9123", "--log-requests"]
    )
    assert (args.host, args.port, args.log_requests) == ("0.0.0.0", 9123, True)


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


def test_explain_command_traces_the_top_hit(capsys):
    assert (
        main(["--corpus", CORPUS, "explain", "what does the b parameter control"]) == 0
    )
    out = capsys.readouterr().out
    assert "fusion (rrf_k=60)" in out
    assert "contrib" in out
    assert "passage:" in out


def test_explain_accepts_a_document_name_with_or_without_its_extension(capsys):
    assert main(["--corpus", CORPUS, "explain", "rank fusion", "--doc", "rank-fusion"]) == 0
    bare = capsys.readouterr().out
    assert (
        main(["--corpus", CORPUS, "explain", "rank fusion", "--doc", "rank-fusion.md"])
        == 0
    )
    assert capsys.readouterr().out == bare


def test_explain_reports_an_unknown_document_by_listing_the_known_ones(capsys):
    assert main(["--corpus", CORPUS, "explain", "anything", "--doc", "nope"]) == 1
    out = capsys.readouterr().out
    assert "no document named" in out and "rank-fusion" in out


def test_explain_rejects_an_out_of_range_chunk(capsys):
    assert main(["--corpus", CORPUS, "explain", "bm25", "--chunk", "100000"]) == 1
    assert "out of range" in capsys.readouterr().out


def test_explain_rank_matches_the_search_ordering(capsys):
    main(["--corpus", CORPUS, "search", "reciprocal rank fusion", "-k", "2"])
    search_out = capsys.readouterr().out
    second_doc = search_out.splitlines()[2].split("[")[1].split("]")[0]
    assert (
        main(["--corpus", CORPUS, "explain", "reciprocal rank fusion", "--rank", "2"])
        == 0
    )
    out = capsys.readouterr().out
    assert out.startswith("#2  [{}]".format(second_doc))


def test_explain_refuses_a_rank_deeper_than_the_result_list(capsys):
    assert (
        main(["--corpus", CORPUS, "explain", "zzzqqq nonexistent", "--rank", "3"]) == 1
    )
    assert "fewer than 3 results" in capsys.readouterr().out


def test_fusion_weights_flag_changes_the_ranking(capsys):
    """A weight the engine ignored would be a silent no-op, so assert it bites."""
    assert (
        main(
            [
                "--corpus",
                CORPUS,
                "--ngram",
                "4",
                "--fusion-weights",
                "ngram=1000",
                "search",
                "how does product quantisation work",
                "-k",
                "3",
            ]
        )
        == 0
    )
    weighted = capsys.readouterr().out
    assert (
        main(
            [
                "--corpus",
                CORPUS,
                "--ngram",
                "4",
                "search",
                "how does product quantisation work",
                "-k",
                "3",
            ]
        )
        == 0
    )
    assert weighted != capsys.readouterr().out


def test_uniform_fusion_weights_flag_is_a_no_op(capsys):
    args = ["--corpus", CORPUS, "search", "reciprocal rank fusion", "-k", "3"]
    assert main(args) == 0
    plain = capsys.readouterr().out
    assert main(["--fusion-weights", "bm25=2,vector=2"] + args) == 0
    assert capsys.readouterr().out == plain


def test_fusion_weights_rejects_a_malformed_pair():
    with pytest.raises(SystemExit, match="name=value"):
        main(["--corpus", CORPUS, "--fusion-weights", "bm25", "stats"])


def test_fusion_weights_rejects_a_non_numeric_weight():
    with pytest.raises(SystemExit, match="not a number"):
        main(["--corpus", CORPUS, "--fusion-weights", "bm25=heavy", "stats"])


def test_fusion_weights_rejects_an_unknown_list(capsys):
    with pytest.raises(ValueError, match="unknown fusion list"):
        main(["--corpus", CORPUS, "--fusion-weights", "colbert=2", "stats"])
