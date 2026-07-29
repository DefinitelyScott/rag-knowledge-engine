# Retrieval-augmented generation

Retrieval-augmented generation combines a retriever over an external corpus
with a generator that conditions its output on what was retrieved. The pattern
was named by Lewis and colleagues in 2020. The motivation is that a language
model's parameters are a lossy, frozen store of knowledge: they cannot be
updated cheaply, they cannot cite anything, and they produce fluent text
regardless of whether the underlying fact was ever learned.

A RAG pipeline has four stages. Ingestion loads source documents and splits
them into chunks. Indexing builds whatever structures retrieval needs, sparse,
dense or both. Retrieval takes a user question and returns the top passages.
Generation places those passages in the prompt and asks the model to answer
from them, with instructions to cite sources and to decline when the context
does not support an answer.

The benefits are concrete. Knowledge can be updated by re-indexing rather than
retraining. Answers can carry citations, which makes them checkable. And the
system can be pointed at private data that no public model was trained on.

The failure modes are equally concrete, and most of them are retrieval
failures wearing a generation costume. If the right passage is never retrieved,
no prompt engineering can recover it; the model will either refuse or invent.
If too many passages are stuffed into the context, the relevant one can be
crowded out or lost in the middle. And a model can still contradict its
sources, which is why faithfulness, meaning whether each claim is supported by
the retrieved context, is measured separately from answer correctness.

A useful discipline is to evaluate the retriever independently of the
generator. Retrieval metrics such as recall@k and MRR set the ceiling on
end-to-end quality: the generator can only be as right as the evidence it was
given. Fixing retrieval first is almost always cheaper than compensating for it
downstream.
