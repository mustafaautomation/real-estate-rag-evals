import numpy as np

from harbor_rag.chunking import chunk_document
from harbor_rag.embeddings import HashedTfidfEmbedder
from harbor_rag.retrieval import BottomRetriever, FaissIndex, vector_literal
from harbor_rag.types import Document


def test_embeddings_are_deterministic_and_separate_topics() -> None:
    embedder = HashedTfidfEmbedder(dim=128)
    texts = [
        "Monthly rent at 18 Lark Street is $2,450.",
        "Monthly rent at 18 Lark Street is $2,450.",
        "Charcoal grills are not allowed on North Pier balconies.",
    ]
    embedder.fit(texts)
    vectors = embedder.embed_documents(texts)
    same = float(np.dot(vectors[0], vectors[1]))
    different = float(np.dot(vectors[0], vectors[2]))
    assert same > 0.99
    assert different < same


def test_faiss_retrieves_the_lark_listing_first() -> None:
    lark = Document(
        doc_id="listing-18-lark",
        title="18 Lark Street",
        body="Monthly rent is $2,450. Dogs are not allowed.",
    )
    pier = Document(
        doc_id="listing-44-north-pier",
        title="44 North Pier",
        body="Monthly rent is $1,975. Heat and water are included in the rent.",
    )
    chunks = chunk_document(lark, 2000, 100) + chunk_document(pier, 2000, 100)
    embedder = HashedTfidfEmbedder(dim=128)
    embedder.fit([chunk.text for chunk in chunks])
    index = FaissIndex(chunks, embedder)
    top = index.retrieve("What is the monthly rent at 18 Lark Street?", k=1)
    assert top[0].chunk.doc_id == "listing-18-lark"
    bottom = BottomRetriever(index).retrieve("What is the monthly rent at 18 Lark Street?", k=1)
    assert bottom[0].chunk.doc_id != "listing-18-lark"


def test_vector_literal_format() -> None:
    literal = vector_literal([0.1, -0.25])
    assert literal == "[0.10000000,-0.25000000]"
