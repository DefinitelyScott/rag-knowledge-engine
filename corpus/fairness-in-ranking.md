# Fairness in ranking: exposure, attention and its limits

Ranking effectiveness is usually measured by how well relevant items are placed
near the top. Fairness asks a different question: who gets seen. When the items
being ranked are people or the products of people, such as job candidates,
sellers, creators or news publishers, the ranking decides how much attention
each of them receives, and that attention is the thing of value. A system that
is excellent by nDCG can still concentrate nearly all visibility on a small
group.

The reason is position bias. Users examine results from the top and give far
less attention to lower ranks, so the attention an item gets is not its
relevance but a steeply falling function of its position. Two candidates whose
relevance differs by a hair end up at ranks one and two, and the gap in exposure
between those positions can be much larger than the gap in merit. The same
effect compounds through feedback: items that were shown more collect more
clicks, which raises their estimated relevance, which earns them more exposure.
This is a form of popularity bias, and it is why click data can reinforce an
existing imbalance instead of correcting it.

Several formal definitions exist, and they disagree. Group fairness compares
protected groups, for instance requiring that a group appear in the top k in
proportion to its share of the pool. Zehlike and colleagues proposed FA*IR in
2017, which builds a top-k list and applies a statistical test at each prefix so
that the protected group is not under-represented by more than chance would
explain. Singh and Joachims, in 2018, argued for fairness of exposure: the
expected attention given to a group should be proportional to its average
relevance, with the exposure model taken from the position discount. Biega,
Gummadi and Weikum, also in 2018, focused on individuals and asked that each
item's attention match its relevance, an idea they called equity of attention.

A single ranking cannot split attention evenly, because only one item can be
first. So individual fairness is usually pursued in an amortised way over many
rankings of the same items. A system can rotate who holds the top spots, so
that over repeated queries the cumulative attention tracks cumulative relevance.
Doing so costs something: serving a less relevant item first sacrifices
immediate utility, so the practical question is how much effectiveness to trade
and where. These methods are typically implemented as a re-ranking step or as a
constrained optimisation over a probabilistic ranking.

The limits are real. Group-based methods need group labels, and collecting
sensitive attributes raises its own privacy and legal issues. The right fairness
notion depends on the application and the groups, and no one definition satisfies
all the others at once. Measured relevance itself may carry bias from past
judgments or clicks, so matching exposure to relevance can preserve an unfair
baseline. Fairness metrics therefore complement relevance metrics rather than
replace them, and the choice of definition is a policy decision, not a purely
technical one.
