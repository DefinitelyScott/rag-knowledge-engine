# Knowledge distillation for retrieval models

A cross-encoder reads the query and the passage together and is usually the
most accurate scorer available, but it is too slow to run over a whole
collection. A bi-encoder or a sparse model is fast enough to search everything
but is usually less accurate. Knowledge distillation tries to close part of
that gap: a large, slow teacher produces training signal for a small, fast
student, so the student learns to imitate the teacher without inheriting its
query-time cost.

The idea comes from Hinton, Vinyals and Dean in 2015. A classifier's output
probabilities carry more information than the single correct label. A teacher
that says a picture is a cat, quite possibly a lynx, and certainly not a truck
is telling the student which wrong answers are nearly right. Training the
student on those soft targets, usually with a temperature that flattens the
distribution so the small probabilities are visible, transfers that extra
structure. Sanh and colleagues applied the same recipe to language models in
2019 with DistilBERT, reporting a model about forty percent smaller and sixty
percent faster that kept most of the original's accuracy on benchmark tasks.

Ranking is not classification, and the scores a ranker emits are not
probabilities. The teacher's scores for a query and its candidates have an
arbitrary scale, and a cross-encoder and a bi-encoder do not produce scores on
the same scale at all. Hofstatter and colleagues addressed this in 2020 with
cross-architecture distillation and a loss called Margin-MSE. For each
training triple of a query, a relevant passage and a non-relevant passage, the
student is trained to reproduce the teacher's score difference between the two
rather than the scores themselves. Only the margin has to agree, so the student
is free to use its own scale. They distilled an ensemble of cross-encoders into
faster architectures on the MS MARCO passage collection.

The teacher can also live inside the pipeline instead of beside it. Izacard
and Grave proposed in 2020 to distil from the reader to the retriever in
question answering. The reader attends over the passages it is given, and the
attention it pays to each passage indicates which ones were actually useful for
the answer. Those attention scores become synthetic relevance labels for
training the retriever, so no human judgements of passages are needed. Later
dense retrievers such as RocketQA used a cross-encoder to clean up the training
data, and ColBERTv2 used distillation from a cross-encoder together with
hard negatives mined from earlier models. Distillation and negative mining
therefore tend to be used together, because a teacher's scores say how hard
each mined candidate really is.

Distillation moves cost from query time to training time. The teacher is
queried once per training example, which can be expensive, but the deployed
student searches at its own speed. It also gives denser supervision than
binary labels. A relevance label says only that one passage is better than
another; a teacher's score says by how much.

The limits are plain. The student is generally bounded by its teacher, and the
teacher's mistakes are copied along with its strengths. If the teacher is
unreliable on false negatives, passages that are relevant but unlabelled, the
student can learn to push those down as well. A teacher trained on one domain
transfers its blind spots to student models used in another. And the extra
training pipeline needs more engineering than fine-tuning on labels alone, so
the gain has to be weighed against that cost.

This document was written from recollection of the cited papers and has not
been checked against the originals.
