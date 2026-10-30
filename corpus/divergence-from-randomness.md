# Divergence from randomness

Divergence from randomness, or DFR, is a family of probabilistic weighting
models introduced by Gianni Amati and Keith van Rijsbergen in 2002. Its starting
point is an old observation from Harter's work on the two-Poisson model:
function words are spread over a collection roughly as if by chance, while a
content-bearing word turns up in a few documents far more often than chance
would predict. DFR turns that observation into a weight. The more a term's
frequency in a document diverges from what a random process would produce, the
more the term says about that document.

A DFR model is assembled from three parts, and its name lists them. The first
is the basic randomness model, which gives the probability of seeing the term
tf times in the document if occurrences were scattered at random. Choices
include a Poisson approximation, a Bose-Einstein model, and models built from
document frequency or expected document frequency that behave like IDF. The
informative content of the term is the negative logarithm of that probability:
an improbable count carries many bits.

The second part, the first normalisation or after-effect, tempers that
surprise. Once a term has appeared in a document, further occurrences are less
surprising, so only part of the informative content is credited. Laplace's law
of succession gives a factor of one over tf plus one; a ratio of two Bernoulli
processes gives a factor that also depends on the term's total frequency in the
collection. This factor is where DFR gets the saturation that BM25 builds in
with its k1 parameter, derived here from a probability argument rather than
chosen as a curve.

The third part, the second normalisation, adjusts the raw term frequency for
document length before either of the other parts sees it. Normalisation 2, the
most used, rescales tf by the logarithm of one plus c times the average length
over the document's length, so a term in a short document counts for more. The
free parameter c plays the role BM25 gives to b, and it is the only constant
most DFR models need tuned.

Models are named by the three letters. PL2 combines the Poisson model, Laplace
after-effect and normalisation 2; InL2 swaps in an IDF-style randomness model;
BB2 uses Bose-Einstein with the Bernoulli after-effect. Later work by Amati
produced variants such as DPH that remove the free parameter entirely, which is
attractive when there are no judged queries to tune against.

The framework is implemented in the Terrier platform from the University of
Glasgow, where DFR models served as strong baselines in TREC tracks. On standard
test collections they perform in the same range as a well-tuned BM25 and as
query likelihood with Dirichlet smoothing. The practical case for DFR is less a
consistent win than a clearer account of each component, and a menu of
interchangeable parts that can be tested one at a time.
