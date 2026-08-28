"""Tests for the standard-library HTTP demo.

Two levels on purpose. The payload builders and the query-string validator are
plain functions, so most of the behaviour is tested without a socket at all.
A single end-to-end test then binds a real server on an OS-assigned port and
speaks HTTP to it, because the parts that only exist inside
``BaseHTTPRequestHandler`` - routing, status codes, content types - are not
reachable any other way, and asserting on them by reading the source is not a
test.
"""

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from ragkb.abstain import AbstentionConfig
from ragkb.engine import EngineConfig, RAGEngine
from ragkb.serve import (
    ALLOWED_METHODS,
    MAX_K,
    PAGE,
    ask_payload,
    build_server,
    parse_query_args,
    result_to_dict,
    search_payload,
)

DOCUMENTS = {
    "bm25": (
        "BM25 is a bag of words ranking function. It scores a document by term "
        "frequency saturation and inverse document frequency. The k1 parameter "
        "controls how quickly term frequency saturates, and b controls how "
        "strongly the score is normalised by document length."
    ),
    "vectors": (
        "TF-IDF represents a passage as a sparse vector of weighted terms. "
        "Cosine similarity between two such vectors measures their overlap "
        "independently of length. Dense embeddings replace the sparse weights "
        "with learned coordinates."
    ),
    "fusion": (
        "Reciprocal rank fusion combines several ranked lists without needing "
        "their scores to be comparable. Each list contributes one over a "
        "constant plus the rank, so a passage that several retrievers rank "
        "highly rises above one that a single retriever loves."
    ),
}


@pytest.fixture(scope="module")
def engine() -> RAGEngine:
    return RAGEngine(DOCUMENTS, config=EngineConfig(target_tokens=40, overlap_tokens=10))


@pytest.fixture(scope="module")
def base_url(engine: RAGEngine):
    """A live server on an OS-assigned port, shut down when the module ends."""
    server = build_server(engine, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield "http://{}:{}".format(host, port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def fetch(url: str):
    with urlopen(url, timeout=10) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def fetch_json(url: str):
    status, content_type, body = fetch(url)
    return status, content_type, json.loads(body.decode("utf-8"))


# --- query-string validation -------------------------------------------------


def test_parse_query_args_defaults():
    query, k, method = parse_query_args({"q": ["what is bm25"]}, default_k=5)
    assert query == "what is bm25"
    assert k == 5
    assert method == "hybrid"


def test_parse_query_args_strips_whitespace():
    query, _, _ = parse_query_args({"q": ["  fusion  "]}, default_k=5)
    assert query == "fusion"


@pytest.mark.parametrize("params", [{}, {"q": [""]}, {"q": ["   "]}])
def test_parse_query_args_requires_a_query(params):
    with pytest.raises(ValueError, match="q"):
        parse_query_args(params, default_k=5)


def test_parse_query_args_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        parse_query_args({"q": ["x"], "method": ["magic"]}, default_k=5)


@pytest.mark.parametrize("method", ALLOWED_METHODS)
def test_parse_query_args_accepts_every_cli_method(method):
    _, _, parsed = parse_query_args({"q": ["x"], "method": [method]}, default_k=5)
    assert parsed == method


def test_parse_query_args_clamps_k_to_the_maximum():
    _, k, _ = parse_query_args({"q": ["x"], "k": ["9999"]}, default_k=5)
    assert k == MAX_K


@pytest.mark.parametrize("raw", ["0", "-3"])
def test_parse_query_args_rejects_non_positive_k(raw):
    with pytest.raises(ValueError, match="at least 1"):
        parse_query_args({"q": ["x"], "k": [raw]}, default_k=5)


def test_parse_query_args_rejects_non_numeric_k():
    with pytest.raises(ValueError, match="integer"):
        parse_query_args({"q": ["x"], "k": ["five"]}, default_k=5)


# --- payload shape -----------------------------------------------------------


def test_result_to_dict_omits_stages_that_did_not_run(engine: RAGEngine):
    result = engine.search("how does bm25 score a document", k=1)[0]
    payload = result_to_dict(1, result)
    assert payload["rank"] == 1
    assert payload["doc_id"] == "bm25"
    assert payload["text"]
    # MMR and the cross-encoder are off in this config, so their ranks must be
    # absent rather than present-and-null.
    assert "mmr_rank" not in payload["provenance"]
    assert "rerank_rank" not in payload["provenance"]
    assert payload["provenance"]["bm25_rank"] >= 1


def test_search_payload_reports_ranking_and_redundancy(engine: RAGEngine):
    payload = search_payload(engine, "reciprocal rank fusion", k=3, method="hybrid")
    assert payload["count"] == len(payload["results"])
    assert payload["count"] <= 3
    assert payload["results"][0]["doc_id"] == "fusion"
    assert [r["rank"] for r in payload["results"]] == list(
        range(1, payload["count"] + 1)
    )
    assert 0.0 <= payload["redundancy"] <= 1.0


def test_search_payload_is_json_serialisable(engine: RAGEngine):
    payload = search_payload(engine, "cosine similarity", k=2, method="vector")
    assert json.loads(json.dumps(payload)) == payload


def test_ask_payload_cites_its_sources(engine: RAGEngine):
    payload = ask_payload(engine, "what does the b parameter control", k=3, method="hybrid")
    assert payload["answer"]
    assert "bm25" in payload["citations"]
    assert payload["abstained"] is False
    # The gate is off in this engine, so there is no report to explain.
    assert "answerability" not in payload


def test_ask_payload_surfaces_an_abstention():
    gated = RAGEngine(
        DOCUMENTS,
        config=EngineConfig(
            target_tokens=40, overlap_tokens=10, abstain=AbstentionConfig(threshold=1.0)
        ),
    )
    payload = ask_payload(gated, "what is the capital of Peru", k=3, method="hybrid")
    assert payload["abstained"] is True
    assert payload["citations"] == []
    assert payload["answerability"]


# --- the page ----------------------------------------------------------------


def test_page_never_interpolates_input_into_markup():
    """The XSS argument in the module docstring, asserted rather than claimed."""
    assert "innerHTML" not in PAGE
    assert "document.write" not in PAGE
    assert "textContent" in PAGE


# --- live server -------------------------------------------------------------


def test_root_serves_the_html_page(base_url: str):
    status, content_type, body = fetch(base_url + "/")
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"<title>ragkb demo</title>" in body


def test_stats_endpoint(base_url: str):
    status, content_type, payload = fetch_json(base_url + "/api/stats")
    assert status == 200
    assert content_type.startswith("application/json")
    assert payload["stats"]["documents"] == len(DOCUMENTS)
    assert payload["stats"]["chunks"] >= len(DOCUMENTS)


def test_search_endpoint(base_url: str):
    status, _, payload = fetch_json(base_url + "/api/search?q=rank+fusion&k=2")
    assert status == 200
    assert payload["method"] == "hybrid"
    assert payload["k"] == 2
    assert payload["count"] <= 2
    assert payload["results"]


def test_ask_endpoint(base_url: str):
    status, _, payload = fetch_json(
        base_url + "/api/ask?q=what+is+cosine+similarity&method=vector"
    )
    assert status == 200
    assert payload["method"] == "vector"
    assert payload["citations"]


def test_trailing_slash_routes_the_same(base_url: str):
    status, _, payload = fetch_json(base_url + "/api/search/?q=bm25")
    assert status == 200
    assert payload["results"]


def test_missing_query_is_a_400(base_url: str):
    with pytest.raises(HTTPError) as caught:
        fetch(base_url + "/api/search")
    assert caught.value.code == 400
    assert "q" in json.loads(caught.value.read().decode("utf-8"))["error"]


def test_bad_method_is_a_400(base_url: str):
    with pytest.raises(HTTPError) as caught:
        fetch(base_url + "/api/ask?q=bm25&method=telepathy")
    assert caught.value.code == 400


def test_unknown_route_is_a_404(base_url: str):
    with pytest.raises(HTTPError) as caught:
        fetch(base_url + "/api/nope")
    assert caught.value.code == 404
