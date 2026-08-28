"""A small read-only HTTP demo for the engine, built on the standard library.

The obvious way to demo a retrieval engine is FastAPI or Streamlit, and the
obvious reason not to is that this repository's whole claim is that it has no
dependencies. A demo that makes ``pip install`` a precondition for running the
project quietly retracts that claim, so this server is written on
``http.server`` instead: it is less ergonomic than a framework and it is worth
knowing exactly what that costs.

What it is not: ``ThreadingHTTPServer`` handles one request per thread with no
worker pool, no timeouts beyond the socket's, no authentication, and no rate
limiting. It binds to ``127.0.0.1`` by default for that reason. It is a way to
click through retrieval behaviour and to have something to point at in a demo
- not a serving tier.

The design decisions that follow from "demo, not service":

* **Read-only.** Only ``GET`` is routed. Nothing here mutates the index, so
  there is no request body to validate and no write path to secure.
* **One shared engine.** The engine is fitted once at startup and shared across
  threads. Retrieval only reads the fitted state, so that sharing is safe, and
  it means a request costs a query rather than a re-index.
* **The API returns the provenance, not just the ranking.** Each result carries
  its per-list ranks and fused score, because the interesting thing to show a
  reader is *why* hybrid ranked something first, not merely that it did.
* **The HTML page is static and inserts every dynamic value with
  ``textContent``.** The server therefore never interpolates user input into
  markup, which is the entire XSS surface of a page like this.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from .engine import RAGEngine
from .retriever import METHODS, NGRAM_METHOD, RetrievalResult

#: Methods the API will accept, mirroring the CLI's ``--method`` choices.
ALLOWED_METHODS: Tuple[str, ...] = METHODS + (NGRAM_METHOD,)

#: Upper bound on ``k``. Without it a single request can ask for the whole
#: corpus back as JSON; the demo has no use for that and it is a trivially
#: cheap way to make the endpoint unpleasant.
MAX_K = 25

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def result_to_dict(rank: int, result: RetrievalResult) -> Dict[str, Any]:
    """Serialise one retrieved chunk, keeping the fusion provenance."""
    payload: Dict[str, Any] = {
        "rank": rank,
        "doc_id": result.doc_id,
        "score": round(result.score, 6),
        "method": result.method,
        "text": result.text,
    }
    provenance = {
        "bm25_rank": result.bm25_rank,
        "vector_rank": result.vector_rank,
        "ngram_rank": result.ngram_rank,
        "mmr_rank": result.mmr_rank,
        "rerank_rank": result.rerank_rank,
    }
    # Only report the stages that actually ran: a null-filled object invites the
    # reader to think a disabled stage returned nothing, rather than that it was
    # never asked.
    payload["provenance"] = {
        name: value for name, value in provenance.items() if value is not None
    }
    return payload


def search_payload(
    engine: RAGEngine, query: str, k: int, method: str
) -> Dict[str, Any]:
    """The ``/api/search`` body: the ranking plus its redundancy."""
    results = engine.search(query, k=k, method=method)
    return {
        "query": query,
        "k": k,
        "method": method,
        "count": len(results),
        "results": [
            result_to_dict(rank, result)
            for rank, result in enumerate(results, start=1)
        ],
        # Reported alongside the results rather than on request: the point of
        # MMR is visible only as a before/after on this number.
        "redundancy": round(engine.retriever.redundancy(results), 6),
    }


def ask_payload(engine: RAGEngine, query: str, k: int, method: str) -> Dict[str, Any]:
    """The ``/api/ask`` body: the extractive answer, its citations, its gate."""
    answer = engine.answer(query, k=k, method=method)
    payload: Dict[str, Any] = {
        "query": query,
        "k": k,
        "method": method,
        "answer": answer.text,
        "citations": list(answer.citations),
        "answerer": answer.method,
        "abstained": answer.abstained,
    }
    if answer.answerability is not None:
        payload["answerability"] = answer.answerability.explain()
    return payload


def _positive_int(raw: Optional[str], default: int, maximum: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("k must be an integer")
    if value < 1:
        raise ValueError("k must be at least 1")
    return min(value, maximum)


def parse_query_args(params: Dict[str, List[str]], default_k: int) -> Tuple[str, int, str]:
    """Validate ``q``, ``k`` and ``method`` from a parsed query string.

    Kept separate from the handler so the validation is testable without a
    socket, and so every endpoint rejects the same inputs the same way.
    """
    query = (params.get("q", [""])[0] or "").strip()
    if not query:
        raise ValueError("missing required parameter: q")
    k = _positive_int(params.get("k", [None])[0], default_k, MAX_K)
    method = params.get("method", ["hybrid"])[0] or "hybrid"
    if method not in ALLOWED_METHODS:
        raise ValueError(
            "unknown method: {} (expected one of {})".format(
                method, ", ".join(ALLOWED_METHODS)
            )
        )
    return query, k, method


class DemoHandler(BaseHTTPRequestHandler):
    """Route ``GET`` requests against the engine attached to the server."""

    server_version = "ragkb-demo"
    #: Set by :func:`build_server`; every handler instance reads the same engine.
    engine: Optional[RAGEngine] = None
    #: Silence the per-request stderr log unless the server was told to keep it.
    quiet: bool = True

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if not self.quiet:
            super().log_message(format, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        route = parsed.path.rstrip("/") or "/"
        engine = self.engine
        if engine is None:  # pragma: no cover - build_server always attaches one
            self._send_json(500, {"error": "no engine attached to the server"})
            return
        if route == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/stats":
            self._send_json(200, {"stats": engine.stats()})
            return
        if route in ("/api/search", "/api/ask"):
            default_k = 5 if route == "/api/search" else 4
            try:
                query, k, method = parse_query_args(params, default_k)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            if route == "/api/search":
                self._send_json(200, search_payload(engine, query, k, method))
            else:
                self._send_json(200, ask_payload(engine, query, k, method))
            return
        self._send_json(404, {"error": "no such route: {}".format(parsed.path)})


def build_server(
    engine: RAGEngine,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    quiet: bool = True,
) -> ThreadingHTTPServer:
    """Bind a server serving ``engine``.

    Returned unstarted so callers - including the tests - decide how to run it.
    Passing ``port=0`` asks the OS for a free port, which is what the tests do
    to avoid depending on 8000 being available.
    """
    handler = type(
        "BoundDemoHandler", (DemoHandler,), {"engine": engine, "quiet": quiet}
    )
    return ThreadingHTTPServer((host, port), handler)


def serve(
    engine: RAGEngine,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    quiet: bool = True,
) -> None:  # pragma: no cover - blocking loop
    """Serve ``engine`` until interrupted."""
    server = build_server(engine, host=host, port=port, quiet=quiet)
    bound_host, bound_port = server.server_address[:2]
    # flush=True because stdout is block-buffered when redirected to a file or
    # a pipe, and a server that prints its own URL only after it is killed is
    # worse than one that prints nothing.
    print(
        "serving {} chunks on http://{}:{}".format(
            len(engine.chunks), bound_host, bound_port
        ),
        flush=True,
    )
    print("press Ctrl-C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ragkb demo</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 0 auto;
         max-width: 46rem; padding: 2rem 1rem 4rem; }
  h1 { font-size: 1.3rem; margin: 0 0 .25rem; }
  p.sub { margin: 0 0 1.5rem; opacity: .7; }
  form { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1rem; }
  input[type=text] { flex: 1 1 18rem; padding: .5rem .6rem; font: inherit; }
  select, input[type=number], button { padding: .5rem .6rem; font: inherit; }
  button { cursor: pointer; }
  .answer { border-left: 3px solid currentColor; padding: .1rem 0 .1rem .8rem;
            margin: 1rem 0; white-space: pre-wrap; }
  .cites, .meta { opacity: .7; font-size: .85rem; }
  ol { padding-left: 1.4rem; }
  li { margin-bottom: 1rem; }
  .doc { font-weight: 600; }
  .snippet { white-space: pre-wrap; margin: .3rem 0 0; }
  .err { color: #c0392b; }
</style>
</head>
<body>
<h1>ragkb</h1>
<p class="sub">Dependency-free hybrid retrieval: BM25 + TF-IDF fused with Reciprocal Rank Fusion.</p>

<form id="f">
  <input type="text" id="q" placeholder="Ask the corpus something..." autofocus>
  <select id="method">
    <option value="hybrid">hybrid</option>
    <option value="bm25">bm25</option>
    <option value="vector">vector</option>
  </select>
  <input type="number" id="k" value="5" min="1" max="25" title="k">
  <button type="submit">Ask</button>
</form>

<div id="out"></div>
<p class="meta" id="stats"></p>

<script>
const out = document.getElementById("out");
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

async function get(url) {
  const r = await fetch(url);
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || ("HTTP " + r.status));
  return body;
}

async function run(event) {
  event.preventDefault();
  const q = document.getElementById("q").value.trim();
  if (!q) return;
  const method = document.getElementById("method").value;
  const k = document.getElementById("k").value;
  const qs = "q=" + encodeURIComponent(q) + "&method=" + method + "&k=" + k;
  out.replaceChildren(el("p", "meta", "searching..."));
  try {
    const [ask, search] = await Promise.all([
      get("/api/ask?" + qs), get("/api/search?" + qs)
    ]);
    const frag = document.createDocumentFragment();
    frag.append(el("div", "answer", ask.answer));
    if (ask.citations.length) {
      frag.append(el("p", "cites", "Sources: " + ask.citations.join(", ")));
    }
    if (ask.answerability) {
      frag.append(el("p", "meta", "answerability: " + ask.answerability));
    }
    frag.append(el("p", "meta",
      search.count + " passages, redundancy " + search.redundancy.toFixed(3)));
    const list = el("ol");
    for (const r of search.results) {
      const li = el("li");
      const head = el("p");
      head.append(el("span", "doc", r.doc_id));
      const prov = Object.entries(r.provenance)
        .map(([name, rank]) => name.replace(/_rank$/, "") + " #" + rank).join(", ");
      head.append(el("span", "meta",
        "  score " + r.score.toFixed(5) + (prov ? "  (" + prov + ")" : "")));
      li.append(head);
      li.append(el("p", "snippet", r.text.slice(0, 400)));
      list.append(li);
    }
    frag.append(list);
    out.replaceChildren(frag);
  } catch (error) {
    out.replaceChildren(el("p", "err", String(error.message || error)));
  }
}

document.getElementById("f").addEventListener("submit", run);
get("/api/stats").then(body => {
  const s = body.stats;
  document.getElementById("stats").textContent =
    s.documents + " documents, " + s.chunks + " chunks, " +
    s.vocabulary + " vocabulary terms";
}).catch(() => {});
</script>
</body>
</html>
"""
