# Interleaving: comparing two rankers with live clicks

Judged test collections tell you which ranker scored higher on queries someone
labelled in advance. They cannot tell you what real users prefer on the queries
they actually type. Online evaluation measures that directly, from behaviour. The
standard method is the A/B test: send some users to ranker A and others to ranker
B, then compare a click-based metric such as click-through rate, abandonment, or
the time to the first click. It is simple and it measures the real product, but
it is noisy. Users differ from each other far more than two rankers differ, so
the two groups need a great many queries before a small improvement is visible.

Interleaving removes the between-group noise by showing every user a single
list that mixes the two rankings. Each user then acts as their own control, and
the comparison is made within one result page. Joachims proposed the idea in
2003 as balanced interleaving. The merged list is built by walking down both
rankings in parallel, taking the next result from one and then the other,
skipping any document already shown, with a coin flip deciding which ranker goes
first. A click is credited to the ranker whose top results contain the clicked
document, and the ranker with more credit across many queries wins the
comparison.

Radlinski, Kurup and Joachims showed in 2008 that balanced interleaving has a
bias. If two rankings are almost the same and differ only by a shift of one
position, one of them can collect more credit than the other even when users
click at random. They proposed team-draft interleaving, named after picking
teams in a playground. In each round a coin flip decides which ranker picks
first, and each picks its highest-ranked document not yet in the list. Every
slot is therefore owned by exactly one ranker, and a click counts for the owner
of the slot. Under random clicking, each ranker wins equally often. Later
variants changed how the merged list is built. Probabilistic interleaving, by
Hofmann and colleagues, draws each slot from a distribution that favours
higher-ranked documents, which lets documents outside both top lists appear.
Optimised interleaving, by Radlinski and Craswell, chooses among all valid
merges so that random clicking is unbiased and the comparison is as sensitive
as it can be. Multileaving, by Schuth and colleagues, extends the idea to
comparing several rankers in one list.

The reason to accept this complexity is sensitivity. Chapelle and colleagues
ran a large validation on a commercial web search engine in 2012. They found
that interleaving agreed with the preference direction given by offline metrics
and by A/B tests, and that it detected differences with substantially fewer
queries than comparing click metrics between two user groups. Search teams
therefore tend to use interleaving as a fast filter on many candidate rankers,
and keep the slower A/B test for the few that survive.

The method has limits worth knowing. It gives only a relative preference between
two rankers and says nothing about absolute quality. It inherits every flaw of
clicks: users click on results near the top for reasons unrelated to relevance,
so a ranker can win by being more attractive rather than more useful, and a
click on a good snippet may mean the page itself was not read. It cannot judge
changes that are not a ranking, such as a new page layout or a new answer box.
Both rankers must also be run on every query, which doubles the ranking cost
for the duration of the test. When live traffic is not available or not safe to
experiment on, counterfactual evaluation is the alternative. It reweights
clicks logged under an earlier ranker using inverse propensity scores, an
estimate of how likely each result was to be examined, so that the logs can score
a ranker that never ran.
