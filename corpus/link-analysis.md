# Link analysis: PageRank, HITS and anchor text as evidence about a page

Every scoring function elsewhere in a text retriever reads the words of the
document being scored. The web added a second kind of evidence that is not in
the document at all: other pages point to it. A hyperlink is an author's
decision that the target is worth a reader's attention, so the link graph can be
read as millions of small, independent endorsements. Link analysis is the
family of methods that turns that graph into a score.

PageRank, described by Brin and Page in 1998 as part of the original Google
design, models a random surfer. At each step the surfer either follows one of
the current page's outgoing links, chosen uniformly, or, with a fixed
probability, gets bored and jumps to a page chosen at random from the whole web.
A page's PageRank is the long-run fraction of time the surfer spends on it.
Formally that is the stationary distribution of a Markov chain, which is the
principal eigenvector of the transition matrix, and it is computed by power
iteration: start from a uniform vector and repeatedly multiply by the matrix
until it stops changing. The probability of following a link, the damping
factor, is usually set to 0.85. The random jump is not decoration. Without it,
a surfer can be trapped in a group of pages that link only to each other, and
pages with no outgoing links at all, dangling nodes, leak probability out of
the chain. Teleportation makes the chain irreducible and aperiodic, which is
what guarantees a unique stationary distribution for the iteration to converge
to. The result is a single number per page, computed once, offline, regardless
of any query.

Kleinberg's HITS algorithm, from the same period, is query-dependent instead.
It starts from a root set of pages returned by an ordinary text search, expands
it with the pages they link to and the pages linking to them, and scores only
that base set. Each page gets two scores. A good authority is pointed to by
many good hubs, and a good hub points to many good authorities; the two
definitions are applied alternately until they settle, which converges to the
principal eigenvectors of A-transpose-A and A-A-transpose for the adjacency
matrix A. Because the neighbourhood is built at query time, HITS costs work per
query, and it suffers from topic drift: if the base set contains a tightly
interlinked cluster on a neighbouring subject, that cluster can capture both
scores. Lempel and Moran's SALSA, in 2000, replaced the mutual reinforcement
with a random walk over the same bipartite hub-authority graph, which makes it
less susceptible to that capture.

A uniform jump treats every page as an equally good place to restart. Changing
the jump distribution changes what the scores mean. Haveliwala's topic-sensitive
PageRank, in 2002, precomputes one vector per broad topic by teleporting only to
pages in that topic, then blends the vectors at query time according to how
well the query matches each topic. Personalised PageRank teleports to one
user's bookmarks or to a single seed page, which turns the same computation
into a measure of proximity in the graph.

Anchor text, the clickable words of a link, is often the most useful thing the
graph contributes. McBryan's World Wide Web Worm indexed it in 1994, and the
1998 Google paper attached anchor text to the page it points to rather than the
page it sits on. Other people's descriptions of a page are frequently better
than the page's own: a company's home page may never state what the company is
called in the form people search for, while thousands of links to it do. Anchor
text also lets a system index pages it has never fetched, along with images and
other content that carries no words of its own. Craswell, Hawking and Robertson
showed in 2001 that anchor text is especially effective for finding a named
site or home page, a navigational task where the page's body text is a poor
match.

Combining a static score with a query-dependent one is the practical problem.
Link scores carry no information about topic, so they act as a prior: in the
language-modelling framing, a document's prior probability multiplies its query
likelihood, and Kraaij, Westerveld and Hiemstra showed in 2002 that priors from
URL form and link counts substantially improve entry-page search. Craswell and
colleagues argued in 2005 that the raw values need transforming before they are
added to a BM25 score, because PageRank is heavily skewed and its scale has no
relation to the text score's. In learning-to-rank systems PageRank, in-degree
and URL depth are simply features among hundreds.

Any score derived from endorsements invites forged endorsements. Link farms,
networks of pages that exist only to point at a target, inflate PageRank
cheaply. TrustRank, by Gyöngyi, Garcia-Molina and Pedersen in 2004, propagates
trust outward from a small seed set of pages reviewed by hand, on the
observation that good pages seldom link to spam. In 2005 the major engines
agreed on the rel="nofollow" attribute, which lets a site mark links, such as
those in comments, that it does not vouch for, so they pass no score.
