# Faceted search and metadata filtering

A ranked list answers a question when the person can state it in words. It is a
poor tool when they are exploring: browsing a catalogue of laptops, a library of
reports, or a store of support tickets, where the goal is to narrow a large set
by properties such as price band, year, author, product category or language.
Faceted search adds those properties to the interface. Each document carries
structured metadata, and the system shows, beside the results, the values that
metadata takes together with how many results have each value.

The design draws on the faceted classification used in libraries and was
developed for search interfaces by researchers such as Marti Hearst and her
colleagues in the Flamenco project. Its central idea is that a facet is an
independent dimension, so a collection can be sliced along several of them in
any order, instead of descending through one fixed hierarchy where a user must
guess the right branch first.

The counts next to each value are the part that makes the interface useful, and
the part that costs the most to compute. They must describe the current result
set, not the whole collection, otherwise a user would click a value that
promises forty items and finds none. A common implementation keeps, for every
facet value, a sorted list or bitmap of the documents that have it. Applying
filters becomes set intersection, and the count shown for a value is the size of
the intersection of the result set with that value's list.

The logic of combining selections follows a convention. Choosing several values
within one facet, such as two brands, normally means either one, so they are
joined with OR. Choosing values in different facets, such as a brand and a
price range, means both, so they are joined with AND. A facet's own counts are
usually computed with that facet's filter removed, so that the user can still see
the sibling values they might switch to instead of watching them collapse to
zero.

The same machinery now appears in retrieval-augmented generation, where metadata
filters restrict a search by date, source, tenant or access permission before any
text is ranked. In a vector index this raises a real design choice. Filtering
after the nearest-neighbour search is simple, but if the filter is selective
most of the returned candidates are discarded and the system may return fewer
than the requested number of results, or none. Filtering before the search
guarantees a full result list, but it fights the index structure: a graph
index built over every vector can be left with disconnected regions once most
nodes are excluded, which hurts recall or forces a fall back to an exact scan.
Implementations pick between the two by estimating how selective a filter is.

Metadata is only as good as its extraction. Facets built from inconsistent
values, with one author written three ways or dates in mixed formats, produce
cluttered lists and counts that split one group into several. Cleaning and
normalising metadata is usually a larger part of the work than building the
interface. Facets also need a limit: a field with thousands of distinct values,
such as a person's name, cannot be shown in full and is better served by a
search box than by a list.
