# Time and freshness: ranking when the age of a document matters

Most ranking models treat a collection as a bag of documents with no clock. For
some queries that is wrong. Someone searching for the result of a match, the
status of a flight or the latest release of a library wants a page written
recently, and an excellent but year-old page is the wrong answer. For others,
such as the boiling point of water, age is irrelevant, and a model that
favoured recent pages would only add noise. Temporal ranking is the problem of
telling these cases apart and acting on it.

The first half is deciding whether a query is time-sensitive. Jones and Diaz
described in 2007 a way to do it from temporal profiles: take the publication
dates of the documents retrieved for the query, or of those a user clicked, and
look at the shape of the distribution. A query whose results cluster in a short
window is temporally unambiguous, one with several separate clusters is
temporally ambiguous because it names a recurring event, and one with a flat
spread is atemporal. A sudden surge in how often a query is issued is another
signal. Kleinberg's burst-detection model, from 2002, treats a stream as moving
between a low-rate and a high-rate state and finds the intervals spent in the
high state, which is how a newsworthy query can be recognised within hours.

The second half is acting on it. Li and Croft proposed in 2003 a time-based
language model that places a prior on documents which decays exponentially with
age, so that a recent document starts with a head start and an old one with a
handicap. The decay rate is a parameter, and it should differ by query: a rate
fitted to breaking news would bury everything written last month. Efron and
Golovchenko studied in 2011 how to estimate the temporal behaviour of a query
from its own results, and Dong and colleagues described in 2010 a system that
learned recency as part of ranking, using features of age alongside relevance
features so that a learning-to-rank model could weigh them together instead of
a hand-set decay.

Underneath both halves sits the question of what the date of a document is.
Crawled pages carry a last-modified header that is often missing, set by the
server's clock to the moment of the request, or changed by a trivial edit such
as a new sidebar, so it cannot be trusted. Engines combine several clues: dates
printed in the page, dates in the address, the time a link to the page first
appeared, and the first time a crawler saw the content. Each is noisy, and a
document that is republished may look newly created when its content is old.

Freshness also costs infrastructure. A page published a minute ago can only be
returned if the index already contains it, which pushes towards small
incrementally updated indexes searched alongside the large static one, with
results merged at query time. Cached answers, which assume the ranking is
stable, then have to be refreshed or invalidated sooner for the queries that are
changing fastest.

Evaluation inherits the difficulty. A relevance judgement made today for a
time-sensitive query is wrong tomorrow, so test collections for these queries
carry timestamps and the judgements must be repeated, which makes them more
expensive and less reusable than judgements for stable topics.
