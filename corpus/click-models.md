# Click models: reading relevance out of logs that were never neutral

Clicks are the only relevance signal that arrives in quantity and costs
nothing to collect, which is why every large system learns from them and why
every large system has to correct them first. The correction is needed because
a click log is not a sample of user opinion about documents. It is a record of
what users did with the list one particular ranker chose to show them, and the
ranker's preferences are baked into the data at two levels: which documents
appeared at all, and in what order.

The standard framing is the examination hypothesis. A document is clicked only
if the user both examined it and found it attractive, so the click probability
factors into a term that depends on the position and a term that depends on the
document. Fitting the position term is what "de-biasing" means: once you have an
estimate of how likely a result at rank seven is to be looked at, a click there
counts for more than a click at rank one.

Models differ in how they describe the examining. The cascade model, from
Craswell and colleagues in 2008, assumes the user reads strictly top-down and
stops at the first click, which derives position bias from a mechanism rather
than fitting it as an arbitrary curve. Its limitation is structural: it can only
describe sessions with exactly one click, and says nothing about abandoned
searches or sessions with several. The dynamic Bayesian network of Chapelle and
Zhang in 2009 extends the chain past the click by adding a satisfaction
variable, which lets it separate a document that was attractive enough to draw
the click from one that actually answered the question - a distinction the
click alone cannot make, since both produce the same log line.

Estimating the examination term needs variation the deployed ranker would not
have produced. Randomly swapping results gives a clean estimate and degrades
the live experience; interleaving two rankings, or perturbing pairs of adjacent
positions, buys most of the signal for less harm. Counterfactual learning to
rank, set out by Joachims, Swaminathan and Schnabel in 2017, then uses those
propensities directly: weight each observed click by the reciprocal of the
probability that its position was examined, and the resulting loss is an
unbiased estimate of the one you would have computed with full information. The
price is variance. A click at a position examined five percent of the time
carries a weight of twenty, so estimates are dominated by rare events and the
weights are usually clipped, trading a controlled bias back for stability.

None of this repairs the deeper problem. Position bias concerns documents that
were shown; selection bias concerns documents that were not, and a document the
deployed ranker never surfaced generates no click to reweight. That ceiling is
the same one a pooled test collection has, reached by a different route.
