from python_rag.ingest.ingest_document import ingest_document
from python_rag.retreive.retreive_chunks import get_documents


def multi_chunk_text(sections: int = 12) -> str:
    """Text long enough to split into several 800-token chunks."""
    return "\n\n".join(
        f"Topic {i}: instructions for feature {i}, step {i * 2}, setting {i * 5}. " * 30
        for i in range(sections)
    )


def stored(store) -> dict:
    return store.get(include=["documents", "metadatas"])


def test_ingest_stores_chunks_with_metadata(fake_vector_store, make_txt):
    ingest_document(make_txt(multi_chunk_text(), name="guide.txt"))

    data = stored(fake_vector_store)
    assert len(data["ids"]) > 1
    assert {m["source"] for m in data["metadatas"]} == {"guide.txt"}
    assert {m["tag"] for m in data["metadatas"]} == {".txt"}


def test_reingesting_same_file_does_not_duplicate(fake_vector_store, make_txt):
    # Regression: chunks used to get random IDs, so every ingest added copies.
    path = make_txt(multi_chunk_text())

    ingest_document(path)
    first_ids = stored(fake_vector_store)["ids"]
    ingest_document(path)

    assert sorted(stored(fake_vector_store)["ids"]) == sorted(first_ids)


def test_ingests_pdf(fake_vector_store, make_pdf):
    ingest_document(make_pdf(["Press ZOOM during playback."], name="manual.pdf"))

    data = stored(fake_vector_store)
    assert data["metadatas"][0]["source"] == "manual.pdf"
    assert "Press ZOOM during playback." in data["documents"][0]


def test_query_matching_a_chunk_returns_it_first(fake_vector_store, make_txt):
    ingest_document(make_txt(multi_chunk_text(), name="guide.txt"))
    target = stored(fake_vector_store)["documents"][3]

    results = get_documents(target, top_k=1)

    assert len(results) == 1
    assert results[0].page_content == target
    assert results[0].metadata["source"] == "guide.txt"


def test_top_k_limits_number_of_results(fake_vector_store, make_txt):
    ingest_document(make_txt(multi_chunk_text()))

    assert len(get_documents("feature", top_k=2)) == 2


def test_empty_store_returns_no_documents(fake_vector_store):
    assert get_documents("anything") == []
