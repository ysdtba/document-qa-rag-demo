import pytest
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

import rag_pipeline


class _FakeIndex:
    """Stands in for the vector index; records the top_k it was asked for."""

    docstore = None

    def as_retriever(self, similarity_top_k):
        self.asked_top_k = similarity_top_k
        return BM25Retriever.from_defaults(nodes=_nodes(), similarity_top_k=similarity_top_k)


def _nodes():
    return [TextNode(text=f"chunk {i} revenue profit dividend", id_=f"n{i}") for i in range(30)]


def _pipeline(tmp_path, monkeypatch, *, top_k, mode):
    persisted = BM25Retriever.from_defaults(nodes=_nodes(), similarity_top_k=10)  # index built with TOP_K=10
    persisted.persist(str(tmp_path / "bm25"))
    monkeypatch.setattr(rag_pipeline, "BM25_PATH", str(tmp_path / "bm25"))
    monkeypatch.setattr(rag_pipeline, "TOP_K", top_k)
    monkeypatch.setattr(rag_pipeline, "RETRIEVAL_MODE", mode)
    pipeline = object.__new__(rag_pipeline.RAGPipeline)
    pipeline._index = _FakeIndex()
    pipeline._build_fusion_retriever()
    return pipeline


def test_bm25_uses_current_top_k_not_the_one_saved_at_index_time(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path, monkeypatch, top_k=25, mode="bm25")
    assert pipeline._fusion_retriever.similarity_top_k == 25


def test_hybrid_mode_fuses_both_retrievers_at_current_top_k(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path, monkeypatch, top_k=20, mode="hybrid")
    fusion = pipeline._fusion_retriever
    assert isinstance(fusion, QueryFusionRetriever)
    assert pipeline._index.asked_top_k == 20
    assert [r.similarity_top_k for r in fusion._retrievers] == [20, 20]


def test_vector_mode_uses_only_the_vector_retriever(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path, monkeypatch, top_k=10, mode="vector")
    assert not isinstance(pipeline._fusion_retriever, QueryFusionRetriever)
    assert pipeline._index.asked_top_k == 10


def test_unknown_mode_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        _pipeline(tmp_path, monkeypatch, top_k=10, mode="dense")


def test_no_reranker_keeps_retrieval_order():
    nodes = [NodeWithScore(node=TextNode(text=t), score=s) for t, s in [("a", 0.1), ("b", 0.9), ("c", 0.5)]]
    out = rag_pipeline._NoReranker(top_n=2).postprocess_nodes(nodes, query_str="b")
    assert [n.get_content() for n in out] == ["a", "b"]
