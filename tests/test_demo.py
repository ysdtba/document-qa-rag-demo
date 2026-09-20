from llama_index.core.schema import NodeWithScore, TextNode

from rag_pipeline import RAGPipeline, _MockReranker, _source_heading, _source_preview


def test_mock_demo_extracts_deadline_from_relevant_chunk():
    question = "示例文档中的理赔申请期限是多少？"
    nodes = [
        NodeWithScore(
            node=TextNode(
                text="# 虚构 OTC 交易与异常理赔规则\n本文档是用于展示的示例文档。",
                metadata={"section": "概览"},
            ),
            score=0.03,
        ),
        NodeWithScore(
            node=TextNode(
                text="### 4.2 理赔申请流程\n1. 申请人须在异常发生后 **5 个交易日** 内提交表格。",
                metadata={"section": "理赔申请流程"},
            ),
            score=0.01,
        ),
    ]

    ranked = _MockReranker(top_n=1).postprocess_nodes(nodes, query_str=question)
    answer = RAGPipeline._mock_generate(None, question, ranked)

    assert ranked[0].metadata["section"] == "理赔申请流程"
    assert _source_heading(ranked[0]) == "4.2 理赔申请流程"
    assert _source_preview(ranked[0]).startswith("申请人须在异常发生后 5 个交易日")
    assert "5 个交易日" in answer
    assert "模拟摘录，非 AI 生成" in answer


def test_mock_demo_handles_no_sources():
    assert "没有找到" in RAGPipeline._mock_generate(None, "未知问题", [])
