# Citations, faithfulness and grounding

An answer without a citation is an assertion. The point of building retrieval
into a generation system is that every claim can be traced back to a passage a
reader can check, and that traceability is what makes the system usable in
settings where being confidently wrong is expensive.

Grounding means the answer is derived from the retrieved context rather than
from the model's parameters. Faithfulness is the measurable version of that
property: for each claim in the answer, is it supported by the supplied
sources? Faithfulness is distinct from correctness. An answer can be factually
true and still unfaithful, because the model recalled it rather than reading
it, and such an answer will fail silently the moment the corpus contradicts the
model's training data.

Citation accuracy is the narrower question of whether the specific reference
attached to a claim actually contains that claim. Systems commonly fail here in
a particular way: they cite the first retrieved passage for everything, or
attach a plausible-looking source to a sentence that came from elsewhere.
Checking citation accuracy requires attributing individual sentences, not whole
answers, to sources.

An extractive answerer avoids the problem by construction. Because it only ever
returns sentences copied verbatim from retrieved passages, every sentence has a
known source and cannot be an invention. The price is fluency and synthesis: it
cannot combine two facts into one sentence, and it cannot rephrase for clarity.
That trade is often worth making for an offline baseline, because it isolates
retrieval quality from generation quality.

For generative answerers, the practical defences are instructing the model to
answer only from the numbered sources, keeping the number of passages small
enough that each can be attended to, and requiring an explicit refusal when the
context is insufficient. Each of those is testable, and none of them is
guaranteed.
