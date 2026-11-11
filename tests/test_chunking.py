from harbor_rag.chunking import chunk_document
from harbor_rag.types import Document


def test_short_document_is_one_chunk() -> None:
    document = Document(doc_id="listing", title="18 Lark Street", body="Monthly rent is $2,450.")
    chunks = chunk_document(document, max_chars=2000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "listing"
    assert chunks[0].chunk_id == "listing-0"
    assert "Monthly rent is $2,450." in chunks[0].text
    assert chunks[0].text.startswith("18 Lark Street.")


def test_long_paragraph_overlaps() -> None:
    body = "word " * 80
    document = Document(doc_id="long", title="Long", body=body.strip())
    chunks = chunk_document(document, max_chars=120, overlap=30)
    assert len(chunks) >= 2
    first_body = chunks[0].text.split("\n\n", 1)[1]
    second_body = chunks[1].text.split("\n\n", 1)[1]
    assert first_body[-30:] == second_body[:30]


def test_blank_paragraphs_are_dropped() -> None:
    document = Document(
        doc_id="blank", title="Blank", body="First paragraph.\n\n\n\nSecond paragraph."
    )
    chunks = chunk_document(document, max_chars=2000, overlap=20)
    assert len(chunks) == 1
    assert "First paragraph." in chunks[0].text
    assert "Second paragraph." in chunks[0].text
