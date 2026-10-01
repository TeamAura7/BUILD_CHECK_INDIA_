from __future__ import annotations

from backend.rag.ingestion.chunker import chunk_pages


SAMPLE_PAGE_TEXT = """Regulation 5. Front Setback for Residential Plots
For a residential building on a plot with area exceeding 200 square
metres, the minimum front setback shall be 3.0 metres.

Regulation 6. Rear Setback for Residential Plots
The minimum rear setback for a residential building shall be 2.0 metres.
"""


def _page(text=SAMPLE_PAGE_TEXT):
    return {
        "text": text,
        "page_num": 1,
        "source_file": "synthetic_bbmp_byelaws.txt",
        "file_path": "/tmp/synthetic_bbmp_byelaws.txt",
        "doc_type": "DCR",
        "city": "Bengaluru",
        "municipality": "BBMP",
    }


def test_chunks_split_on_clause_boundaries():
    chunks = chunk_pages([_page()], max_tokens=512, overlap_tokens=50)
    assert len(chunks) == 2
    assert chunks[0]["clause_ref"].startswith("Regulation 5.")
    assert chunks[1]["clause_ref"].startswith("Regulation 6.")


def test_chunk_ids_are_scoped_by_municipality_and_source():
    chunks = chunk_pages([_page()], max_tokens=512, overlap_tokens=50)
    for c in chunks:
        assert c["chunk_id"].startswith("BBMP::synthetic_bbmp_byelaws.txt::")
        assert c["municipality"] == "BBMP"


def test_chunk_text_is_self_contained_with_heading():
    chunks = chunk_pages([_page()], max_tokens=512, overlap_tokens=50)
    assert "Regulation 5. Front Setback" in chunks[0]["text"]
    assert "3.0 metres" in chunks[0]["text"]


def test_no_boundaries_falls_back_to_whole_page_chunk():
    plain = "This page has no clause markers at all, just prose text about setbacks."
    chunks = chunk_pages([_page(plain)], max_tokens=512, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0]["clause_ref"] == "UNCATEGORISED"


def test_chunk_ids_are_unique_across_pages_lacking_clause_markers():
    """Two different pages of the same document with no detectable clause
    boundary (e.g. a preface page and an appendix page) must not collide on
    chunk_id -- hybrid_retriever deduplicates fused results by this id, so a
    collision would silently drop one page's chunk and could misattribute a
    citation to the wrong page."""
    pages = [
        _page("This is a preface page with no clause markers at all here.") | {"page_num": 1},
        _page("This is an appendix page also with no clause markers here.") | {"page_num": 50},
    ]
    chunks = chunk_pages(pages, max_tokens=512, overlap_tokens=50)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c["clause_ref"] == "UNCATEGORISED" for c in chunks)


def test_long_clause_is_sub_split_with_overlap():
    long_para = " ".join(["word"] * 800)  # ~800 tokens at 1 word ~ 1 token approx
    text = f"Regulation 12. A Very Long Clause\n\n{long_para}\n\n{long_para}"
    chunks = chunk_pages([_page(text)], max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c["clause_ref"].startswith("Regulation 12.")
