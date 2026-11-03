# Table retrieval

Table retrieval is the task of finding, for a query, the table that answers it:
a table of country populations for a question about population, or a table of
election results for a question about a vote. Tables are everywhere in
documents and on the web, and they hold much of the structured, factual
content that a passage retriever handles badly, because a table's meaning lives
in its layout as well as its words.

A table is not a short document. Its cells are mostly numbers, names and
abbreviations with little connecting prose. What carries meaning is the
arrangement: column headers say what each column means, a row header or first
column usually names the entity, and a caption or the surrounding page says
what the whole table is about. Flattening the cells into a single string
throws that arrangement away. The number 1984 in a cell could be a year, a
count or a code, and only its header says which.

The first practical step is therefore to treat a table as several fields rather
than one body. The caption, the page title, the section heading, the column
headers and the cell text are indexed separately and weighted differently, in
the way fielded retrieval treats a title against a body. Work on ad hoc table
retrieval by Shuo Zhang and Krisztian Balog in 2018 studied features of this
kind, and found that evidence from the caption and the headers carried much of
the signal. A match on a header is strong evidence the table has the right
columns, while a match in a cell shows only that a value was mentioned.

Linearisation is the bridge to neural models. A table is serialised into a token
sequence, for example row by row with separator tokens between cells, so that a
text encoder can embed it. This is lossy, since a very large table does not fit
in an encoder's input, and the serialisation order imposes a reading order the
table never had. Models designed for tables add structure back: TAPAS, from
Google Research in 2020, adds learned embeddings for the row and the column of
each cell to a transformer, so the model can tell that two cells share a
column. Its first use was answering questions over a given table, and the same
idea was extended to open-domain retrieval, where a dense retriever picks
candidate tables from a large collection for a reader to use.

Numbers are the hard part for any text-matching method. A query such as "which
countries have a population above fifty million" asks for a comparison over a
column, not a string match, and no inverted index can answer it from the words
alone. Retrieval can only narrow the field to tables that have a population
column, and a second stage, such as question answering over the retrieved
table or generating a structured query against it, has to do the filtering and
aggregation. Retrieval and answering are best evaluated separately, because a
system can fetch the right table and still read the wrong cell.

Large tables raise a size problem for retrieval-augmented generation. Putting an
entire table into a prompt wastes the context window and can bury the answer
row. Common remedies are to retrieve the table by its header and caption, and
then select only the rows and columns relevant to the question, or to split a
long table into row groups that each repeat the header, so that every chunk
remains interpretable on its own.
