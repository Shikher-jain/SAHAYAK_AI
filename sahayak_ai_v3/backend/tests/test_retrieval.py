import pytest
import numpy as np
from sahayak_ai_v3.backend.core.models import RetrievedDocument
from sahayak_ai_v3.backend.services.retrieval.fusion import reciprocal_rank_fusion
from sahayak_ai_v3.backend.services.retrieval.sparse import BM25SparseRetriever

@pytest.fixture
def sample_docs():
    return [
        RetrievedDocument(id="1", modality="text", content="Machine learning is fascinating", source="book", metadata={"source": "book"}),
        RetrievedDocument(id="2", modality="text", content="Artificial intelligence is the future", source="article", metadata={"source": "article"}),
        RetrievedDocument(id="3", modality="text", content="Data science involves statistics", source="blog", metadata={"source": "blog"}),
    ]

def test_reciprocal_rank_fusion(sample_docs):
    # Simulate three different rankings
    dense_results = [sample_docs[0], sample_docs[1]]
    sparse_results = [sample_docs[1], sample_docs[2]]
    graph_results = [sample_docs[0]]

    # rrf_score = sum(1 / (k + rank))
    # For doc 1 (id="2"): 
    #   dense_rank = 1 (idx 1), score = 1/62
    #   sparse_rank = 0 (idx 0), score = 1/61
    #   Total: 1/62 + 1/61
    # For doc 0 (id="1"):
    #   dense_rank = 0, score = 1/61
    #   graph_rank = 0, score = 1/61
    #   Total = 2/61
    # For doc 2 (id="3"):
    #   sparse_rank = 1, score = 1/62
    
    k = 60
    fused = reciprocal_rank_fusion(dense_results, sparse_results, graph_results, k=k)
    
    assert len(fused) == 3
    # Check if doc 0 has highest score (2/61)
    assert fused[0].id == "1"
    assert fused[1].id == "2"
    assert fused[2].id == "3"

    assert fused[0].rrf_score == (1.0/61 + 1.0/61)
    assert fused[1].rrf_score == (1.0/62 + 1.0/61)
    assert fused[2].rrf_score == (1.0/62)

@pytest.mark.asyncio
async def test_bm25_retriever_empty():
    retriever = BM25SparseRetriever(max_docs=10)
    # If no documents, should return empty
    results = await retriever.retrieve("test query", top_k=5)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_bm25_retriever_with_docs():
    retriever = BM25SparseRetriever(max_docs=10)
    
    # Manually inject docs for testing
    retriever.documents = [
        {"id": "doc1", "payload": {"content": "apples and oranges are fruits"}},
        {"id": "doc2", "payload": {"content": "cats and dogs are pets"}},
        {"id": "doc3", "payload": {"content": "apples are red and dogs bark"}},
    ]
    
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [doc["payload"]["content"].lower().split() for doc in retriever.documents]
    retriever.bm25 = BM25Okapi(tokenized_corpus)
    
    results = await retriever.retrieve("apples fruits", top_k=2)
    assert len(results) >= 1
    assert results[0].id == "doc1"
    assert results[0].sparse_score > 0

@pytest.mark.asyncio
async def test_graph_retriever_empty():
    from sahayak_ai_v3.backend.services.retrieval.graph import Neo4jGraphRetriever
    retriever = Neo4jGraphRetriever()
    # "unknown entity" shouldn't exist in our mock/empty DB
    results = await retriever.retrieve("completely unknown query xyz123", top_k=5)
    assert len(results) == 0
