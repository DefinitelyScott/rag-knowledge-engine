# Multi-hop retrieval

Most retrieval evaluation assumes a question can be answered by finding one
passage. A large class of real questions cannot. "Which university employed the
author of the paper that introduced MaxSim?" needs a first passage to name the
author and a second, which shares no vocabulary with the original question, to
place them. The entity that connects the two is called the bridge, and a
single-shot retriever fails on such questions for a structural reason rather
than a tuning one: the words needed to find the second passage are not in the
query yet. HotpotQA, released by Yang and colleagues in 2018, was built to make
this failure mode measurable, and it remains the standard reference point.

The fix is to retrieve more than once, using what the first round returned to
write the next query. Decomposition does this explicitly. Self-Ask, from Press
and colleagues in 2022, has the model state a follow-up question, answer it
from retrieved text, and repeat until the original question can be settled;
the same paper names the underlying problem the compositionality gap, the
observation that models often know both facts separately yet fail to compose
them. Interleaving is the looser variant: IRCoT, from Trivedi and colleagues in
2023, alternates one step of chain-of-thought reasoning with one retrieval
using that step as the query, so the reasoning supplies the vocabulary the
original question lacked. FLARE, from Jiang and colleagues in 2023, triggers
retrieval from the generator's own uncertainty, fetching passages only when the
next sentence is about to be produced with low confidence.

All of these cost more than one retrieval and share one failure mode: errors
compound. If the first hop returns the wrong passage, the second hop is issued
against a query built from a false premise, and the pipeline can produce a
fluent, well-cited answer that is wrong from the second sentence onward. Recall
at the first hop therefore matters more than recall anywhere else, and it is
worth measuring per hop rather than only end to end, because a system that
looks 70 percent accurate overall may be 90 percent accurate on hop one and
falling apart afterwards.

The other cost is latency, which multiplies rather than adds: every hop is a
full retrieval plus a generation step, and the hops are sequential by
construction. That makes the routing decision the important one. Questions that
name a single fact should never enter the loop, and a cheap classifier or a
confidence threshold on the first result set can keep most traffic on the
single-shot path.

The honest way to justify any of this is to hold the corpus fixed and measure
what fraction of the gold set actually requires a second hop. On a collection
where nearly every question is answerable from one passage, multi-hop machinery
adds latency and a new way to be confidently wrong, and buys nothing.
