# Learning to rank: fitting the ordering instead of hand-tuning it

Every ranking function discussed so far carries constants somebody chose. BM25
has `k1` and `b`, reciprocal rank fusion has `k`, a weighted fusion has one
weight per list. Learning to rank replaces that hand-tuning with a supervised
fit: treat each query-document pair as a feature vector, treat the relevance
judgements as labels, and let an optimiser find the combination. The features
are usually the scores the other stages already produce - a BM25 score, a
vector similarity, a document length, a freshness value - so this is less a
rival retriever than a principled way to combine the ones in hand.

The approaches divide by what the loss function sees. A pointwise loss treats
ranking as regression onto a relevance grade and scores each document alone,
which is simple and wrong in a specific way: it spends equal effort separating
two irrelevant documents at rank four hundred as it does separating the best
document from the second best. A pairwise loss instead takes two documents with
different labels and penalises the model when it orders them backwards, which
is closer to the thing being optimised. RankNet, from Burges and colleagues in
2005, is the canonical version and uses a cross-entropy loss over pair
orderings. Joachims had used the same pairwise framing in 2002 to learn a
ranker from clickthrough logs, on the argument that a click tells you a
document beat the ones above it that were skipped, not that it was good in
absolute terms.

Listwise methods look at the whole ranked list at once, which is what the
evaluation metric does. The obstacle is that metrics like nDCG are flat almost
everywhere: swapping two adjacent documents either changes the metric in a jump
or not at all, so the gradient is either zero or undefined and there is nothing
for gradient descent to follow. LambdaRank's trick was to skip the loss
entirely and define the gradient directly, scaling each pairwise update by how
much swapping that pair would move nDCG. LambdaMART applies that gradient
inside gradient-boosted regression trees, and Burges' 2010 retrospective traces
the whole line. Boosted trees remained the strongest choice on this kind of
tabular feature vector for years, well after neural models had taken over
scoring from raw text.

The hard part is not the optimiser, it is the labels. Graded judgements are
expensive, so most production systems learn from clicks, and clicks are biased
towards whatever the deployed system already ranked highly - a feedback loop
where the model is trained on data its own predecessor selected. The features
also fix a ceiling: a learned combination of BM25 and a vector score can only
reweight evidence those two produce, and cannot recover a document neither
stage retrieved.
