"""
文档问答 RAG 检索流水线
================================
职责：将「切块 → 向量检索 → BM25 全文检索 → RRF 融合 → Rerank → LLM 流式生成」
      串联为一条可复用的管道，供 FastAPI 异步接口调用。

1. 为什么做 Hybrid Search？
   - 纯向量检索：语义相近但关键词缺失时表现好，但对精确术语（如「OTC-2024-A1」）容易漏召。
   - 纯 BM25：关键词精确匹配强，但不理解同义表述（如「场外交易」vs「OTC」）。
   - 混合检索 = 取长补短，在金融合规文档场景中召回率显著更高。

2. RRF（Reciprocal Rank Fusion，倒数排名融合）怎么做？
   - 不依赖各检索器原始分数的量纲（向量余弦 0~1，BM25 可能上千），避免归一化难题。
   - 公式：RRF_score(d) = Σ 1 / (k + rank_i(d))，k 通常取 60。
   - 某文档在任一路检索器中排名靠前，融合后综合得分就高。
   - LlamaIndex 的 QueryFusionRetriever(mode="reciprocal_rerank") 内置此逻辑。

3. 为什么需要 Rerank？
   - 混合检索 Top-10 仍可能含噪声；Cross-Encoder 对 (query, chunk) 做精细语义打分。
   - 计算量比 bi-encoder 大，因此只对 Top-10 精排，输出 Top-3 再喂给 LLM，平衡精度与延迟。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.llms import ChatMessage, MessageRole, MockLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

# ── 全局配置 ──────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "10"))           # 混合检索召回数
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))  # Rerank 后精选数
CHROMA_PATH = os.getenv("CHROMA_PATH", "./storage/chroma")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "./storage/docstore")
BM25_PATH = os.getenv("BM25_PATH", "./storage/bm25")
RERANK_MODE = os.getenv("RERANK_MODE", "local").lower()
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "65536"))


def _patch_openai_compat() -> None:
    """兼容 DeepSeek 等非 OpenAI 官方模型名（LlamaIndex 白名单不含这些名字）。"""
    from llama_index.llms.openai import base as openai_base
    from llama_index.llms.openai import utils as openai_utils

    _orig_ctx = openai_utils.openai_modelname_to_contextsize
    _orig_chat = openai_utils.is_chat_model

    def _safe_context_size(model_name: str) -> int:
        try:
            return _orig_ctx(model_name)
        except ValueError:
            return LLM_CONTEXT_WINDOW

    def _safe_is_chat(model: str) -> bool:
        # DeepSeek 等 OpenAI 兼容模型需走 Chat Completions API
        if "deepseek" in model.lower():
            return True
        return _orig_chat(model)

    openai_utils.openai_modelname_to_contextsize = _safe_context_size
    openai_base.openai_modelname_to_contextsize = _safe_context_size
    openai_utils.is_chat_model = _safe_is_chat
    openai_base.is_chat_model = _safe_is_chat


def _init_settings() -> None:
    """初始化 Embedding 与 LLM（LLM 无 Key 时由 stream_chat 走 Mock 分支）。"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
    )
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        _patch_openai_compat()
        Settings.llm = OpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=api_key,
            api_base=os.getenv("OPENAI_BASE_URL"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )
    else:
        # QueryFusionRetriever 初始化时需要 Settings.llm，Mock 模式用 MockLLM 占位
        Settings.llm = MockLLM(max_tokens=256)


def _build_reranker():
    """
    重排序层：对 Top-K 候选逐对计算 query-chunk 相关性，精选 Top-N。
    RERANK_MODE=mock 时使用轻量启发式打分，无需下载 Cross-Encoder，适合快速演示。
    """
    if RERANK_MODE == "mock":
        return _MockReranker(top_n=RERANK_TOP_N)
    return SentenceTransformerRerank(
        model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"),
        top_n=RERANK_TOP_N,
    )


class _MockReranker:
    """Mock Reranker：按 query 与 chunk 的字符重叠率近似排序。"""

    def __init__(self, top_n: int = 3) -> None:
        self.top_n = top_n

    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_str: str | None = None
    ) -> list[NodeWithScore]:
        if not query_str:
            return nodes[: self.top_n]

        def _overlap_score(node: NodeWithScore) -> float:
            q_chars = set(query_str)
            c_chars = set(node.get_content())
            if not q_chars:
                return 0.0
            return len(q_chars & c_chars) / len(q_chars)

        ranked = sorted(nodes, key=_overlap_score, reverse=True)
        for i, n in enumerate(ranked[: self.top_n]):
            n.score = 1.0 - i * 0.1
        return ranked[: self.top_n]


class RAGPipeline:
    """RAG 管道：加载持久化索引，提供 retrieve + stream_chat 能力。"""

    def __init__(self) -> None:
        _init_settings()
        self._index: VectorStoreIndex | None = None
        self._fusion_retriever: QueryFusionRetriever | None = None
        self._reranker = _build_reranker()
        self._load_from_disk()

    # ── 持久化加载 ────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        """从 Chroma + Docstore + BM25 快照恢复索引（由 data_init.py 预先构建）。"""
        docstore_file = Path(DOCSTORE_PATH) / "docstore.json"
        if not docstore_file.exists():
            return

        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        docstore = SimpleDocumentStore.from_persist_dir(str(DOCSTORE_PATH))
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = db.get_or_create_collection("hkex_finance_kb")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

        # from_vector_store 不重新 embed，直接挂载已有向量
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage,
        )
        self._build_fusion_retriever()

    def _build_fusion_retriever(self) -> None:
        """
        构建 Hybrid 融合检索器 = 向量 Retriever + BM25 Retriever → RRF 融合。

        两路召回分别覆盖语义匹配和精确术语，RRF 在不比较原始分数的情况下融合排名。
        """
        if self._index is None:
            return

        # 路 1：向量检索（Dense Retrieval，基于 Embedding 余弦相似度）
        vector_retriever = self._index.as_retriever(similarity_top_k=TOP_K)

        # 路 2：BM25 全文检索（Sparse Retrieval，基于 TF-IDF 变体）
        bm25_path = Path(BM25_PATH)
        if bm25_path.exists():
            bm25_retriever = BM25Retriever.from_persist_dir(str(bm25_path))
        else:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self._index.docstore,
                similarity_top_k=TOP_K,
            )

        # RRF 融合：mode="reciprocal_rerank" 即倒数排名融合（Reciprocal Rank Fusion）
        self._fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=TOP_K,
            num_queries=1,              # 不额外做 Query Expansion，降低延迟
            mode="reciprocal_rerank",   # RRF 融合策略
            use_async=False,
        )

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._fusion_retriever is not None

    # ── 检索 + 精排 ───────────────────────────────────────────────────

    def retrieve_and_rerank(self, question: str) -> list[NodeWithScore]:
        """
        同步检索链路（CPU/IO 密集，在 FastAPI 中会通过 run_in_executor 异步化）：
        Hybrid Top-K → Rerank Top-N
        """
        if not self.is_ready:
            raise RuntimeError("知识库未初始化，请先运行: python data_init.py")

        assert self._fusion_retriever is not None
        # Step 1: 混合检索召回 Top-K（内部已完成 RRF 融合）
        candidates: list[NodeWithScore] = self._fusion_retriever.retrieve(question)
        # Step 2: Cross-Encoder / Mock 精排，输出 Top-N
        return self._reranker.postprocess_nodes(candidates, query_str=question)

    def _build_chat_messages(
        self, question: str, nodes: list[NodeWithScore]
    ) -> list[ChatMessage]:
        context = "\n\n---\n\n".join(
            f"[来源: {n.metadata.get('section', '未知')}]\n{n.get_content()}"
            for n in nodes
        )
        return [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "你是虚构示例文档的问答助手。请严格依据以下上下文回答，"
                    "不要编造。若上下文不足，请明确说明。"
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"【上下文】\n{context}\n\n【用户问题】\n{question}",
            ),
        ]

    # ── SSE 流式生成 ──────────────────────────────────────────────────

    async def stream_chat(self, question: str) -> AsyncGenerator[str, None]:
        """
        异步 SSE 事件生成器 —— 核心目标是降低 TTFT（Time To First Token）。

        【流式输出原理】
        1. 检索阶段（Hybrid + Rerank）仍是一次性完成，通常 200~800ms。
        2. 检索完成后立刻推送 `event: sources`，前端可先展示引用来源。
        3. LLM 生成阶段使用 astream_chat，每产出一个 token 就 yield 一条 SSE，
           用户感知首字延迟 ≈ 检索耗时，而非检索+完整生成耗时。
        4. SSE 格式：`data: {json}\n\n`，客户端用 EventSource / iter_lines 消费。
        5. 最后推送 `data: [DONE]\n\n` 作为结束信号。

        【高并发注意】
        - 检索用 run_in_executor 避免阻塞 asyncio 事件循环。
        - 生产环境可加 Semaphore 限流、检索结果缓存、连接池等。
        """
        loop = asyncio.get_running_loop()

        # ── Phase 1: 异步化阻塞检索，不占用事件循环 ──
        nodes: list[NodeWithScore] = await loop.run_in_executor(
            None, self.retrieve_and_rerank, question
        )

        # ── Phase 2: 推送引用来源（客户端可在 LLM 输出前先渲染 citations）──
        sources_payload = {
            "type": "sources",
            "data": [
                {
                    "section": n.metadata.get("section", ""),
                    "score": round(float(n.score or 0.0), 4),
                    "preview": n.get_content()[:120] + "…",
                }
                for n in nodes
            ],
        }
        yield f"data: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

        # ── Phase 3: LLM 流式 token 推送（DeepSeek 等走 Chat Completions API）──
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        use_real_llm = api_key and not isinstance(Settings.llm, MockLLM)
        if use_real_llm:
            messages = self._build_chat_messages(question, nodes)
            stream = await Settings.llm.astream_chat(messages)
            async for chunk in stream:
                token = chunk.delta or ""
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"
        else:
            # Mock 流式：逐字输出，终端演示「一个字一个字蹦」的效果
            mock_answer = self._mock_generate(question, nodes)
            for char in mock_answer:
                yield f"data: {json.dumps({'type': 'token', 'data': char}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.03)  # 模拟网络/推理延迟，便于肉眼观察流式效果

        yield "data: [DONE]\n\n"

    def _mock_generate(self, question: str, nodes: list[NodeWithScore]) -> str:
        """无 API Key 时的 Mock 回答，基于 Rerank 后的 Top-1 片段摘要。"""
        if not nodes:
            return "抱歉，知识库中未找到相关信息。请先运行 python data_init.py 初始化数据。"
        top = nodes[0]
        section = top.metadata.get("section", "相关条例")
        preview = top.get_content()[:200]
        return (
            f"根据《{section}》相关规定：{preview}… "
            f"（以上回答基于知识库检索结果，Mock 模式仅供演示 SSE 流式输出。）"
        )
