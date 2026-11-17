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
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.llms import ChatMessage, MessageRole, MockLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

# 先加载 .env，再读取下面的模块级配置（否则 .env 里的 TOP_K / RERANK_MODE 等不会生效）
load_dotenv()

# ── 全局配置 ──────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "20"))           # 混合检索召回数
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))  # Rerank 后精选数
CHROMA_PATH = os.getenv("CHROMA_PATH", "./storage/chroma")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "./storage/docstore")
BM25_PATH = os.getenv("BM25_PATH", "./storage/bm25")
RERANK_MODE = os.getenv("RERANK_MODE", "local").lower()          # local | mock | none
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()   # hybrid | vector | bm25
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "65536"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
# DeepSeek V4 默认开启 thinking 模式，此时 temperature 不生效；设为 disabled 才能保证 temperature=0
LLM_THINKING = os.getenv("LLM_THINKING", "").strip().lower()


def thinking_extra_body(value: str) -> dict:
    """LLM_THINKING / JUDGE_THINKING → OpenAI SDK 的 extra_body；留空则不传。"""
    return {"thinking": {"type": value}} if value in {"enabled", "disabled"} else {}

# 拒答固定句式：系统提示要求模型找不到答案时原样输出，评测脚本据此判断是否拒答
NOT_FOUND_SENTENCE = "Not found in the provided annual report excerpts."

SYSTEM_PROMPT = (
    "You answer questions about Hong Kong listed companies' annual reports. "
    "Use only the context excerpts provided. Each excerpt is labelled with company, "
    "fiscal year and PDF page. Cite every fact with the label of the excerpt it came from, "
    "copying the company name exactly, e.g. (<company> FY<year> p.<page>). "
    "Give figures with their unit and currency exactly as reported. "
    f"If the excerpts do not contain the answer, reply exactly: \"{NOT_FOUND_SENTENCE}\" "
    "Do not guess or use outside knowledge. Answer in the language of the question."
)


def effective_config() -> dict[str, Any]:
    """当前生效的检索/生成配置，写进评测结果，便于复现和对比。"""
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return {
        "embed_model": os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        "retrieval_mode": RETRIEVAL_MODE,
        "rerank_mode": RERANK_MODE,
        "rerank_model": os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base") if RERANK_MODE == "local" else None,
        "rerank_device": (os.getenv("RERANK_DEVICE") or "auto") if RERANK_MODE == "local" else None,
        "top_k": TOP_K,
        "rerank_top_n": RERANK_TOP_N,
        "llm_model": os.getenv("LLM_MODEL", "deepseek-chat") if has_key else "mock",
        "llm_base_url": os.getenv("OPENAI_BASE_URL") if has_key else None,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_thinking": LLM_THINKING or None,
        "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
        "table_chunks": os.getenv("TABLE_CHUNKS", "off").strip().lower() in {"1", "on", "true", "yes"},
    }


@dataclass
class AnswerResult:
    """非流式回答（评测用）：文本 + 用量 + 实际返回的模型版本。"""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw_usage: dict = field(default_factory=dict)


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
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
    )
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        _patch_openai_compat()
        Settings.llm = OpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=api_key,
            api_base=os.getenv("OPENAI_BASE_URL"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            temperature=LLM_TEMPERATURE,
            additional_kwargs=(
                {"extra_body": thinking_extra_body(LLM_THINKING)} if thinking_extra_body(LLM_THINKING) else {}
            ),
        )
    else:
        # QueryFusionRetriever 初始化时需要 Settings.llm，Mock 模式用 MockLLM 占位
        Settings.llm = MockLLM(max_tokens=256)


def _build_reranker():
    """
    重排序层：对全部 Top-K 候选逐对计算 query-chunk 相关性并重新排序。
    返回完整排序（top_n=TOP_K），由调用方截取前 RERANK_TOP_N 个送给 LLM；
    这样评测能在同一个排序上算 Hit@5 / MRR@10，Cross-Encoder 的计算量不变。
    RERANK_MODE=mock 时使用轻量启发式打分，无需下载 Cross-Encoder，适合快速演示。
    RERANK_MODE=none 时不重排，直接沿用检索（RRF 融合）后的顺序，作为重排的对照组。
    """
    if RERANK_MODE == "none":
        return _NoReranker(top_n=TOP_K)
    if RERANK_MODE == "mock":
        return _MockReranker(top_n=TOP_K)
    return SentenceTransformerRerank(
        model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"),
        top_n=TOP_K,
        # 留空自动选择（Mac 上是 MPS）；候选很多时 MPS 可能显存不足，可设 RERANK_DEVICE=cpu
        device=os.getenv("RERANK_DEVICE") or None,
    )


def _phrase_overlap_score(question: str, content: str) -> int:
    focused_question = re.sub(
        r"(?:示例|这份|该)?文档(?:中的|里的|中|里|的)?|请问|是多少|是什么|多少|什么|如何",
        "",
        question,
    )
    query = "".join(char.lower() for char in focused_question if char.isalnum())
    text = "".join(char.lower() for char in content if char.isalnum())
    return sum(
        width * width
        for width in (2, 3, 4)
        for start in range(max(0, len(query) - width + 1))
        if query[start:start + width] in text
    )


def _get(obj: Any, key: str) -> Any:
    """同时兼容 dict 和 OpenAI SDK 对象的取值。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def source_label(node: NodeWithScore) -> str:
    """引用标签：年报用「公司 FY年份 p.页码」，其他文档用 section。"""
    meta = node.metadata
    if meta.get("page") is not None:
        who = meta.get("company") or meta.get("ticker") or meta.get("file_name", "")
        year = f" FY{meta['fiscal_year']}" if meta.get("fiscal_year") else ""
        return f"{who}{year} p.{meta['page']}"
    return str(meta.get("section", "未知"))


def _source_heading(node: NodeWithScore) -> str:
    if node.metadata.get("page") is not None:
        return source_label(node)
    first_line = node.get_content().strip().splitlines()[0] if node.get_content().strip() else ""
    match = re.match(r"^#{1,6}\s+(.+)$", first_line)
    if match:
        return match.group(1).strip()
    return str(node.metadata.get("section", "相关片段")).split("/")[-1].strip()


def _source_preview(node: NodeWithScore) -> str:
    lines = [
        re.sub(r"^\s*(?:[-*]\s+|\d+\.\s*)", "", line).replace("**", "").strip()
        for line in node.get_content().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ">"))
    ]
    excerpt = " ".join(lines) or _source_heading(node)
    return excerpt[:120] + ("…" if len(excerpt) > 120 else "")


class _NoReranker:
    """不重排：保留检索器给出的顺序（对照实验用）。"""

    def __init__(self, top_n: int = 10) -> None:
        self.top_n = top_n

    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_str: str | None = None
    ) -> list[NodeWithScore]:
        return nodes[: self.top_n]


class _MockReranker:
    """Mock Reranker：按 query 与 chunk 的短语重叠近似排序。"""

    def __init__(self, top_n: int = 3) -> None:
        self.top_n = top_n

    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_str: str | None = None
    ) -> list[NodeWithScore]:
        if not query_str:
            return nodes[: self.top_n]

        ranked = sorted(
            nodes,
            key=lambda node: _phrase_overlap_score(query_str, node.get_content()),
            reverse=True,
        )
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
            # 持久化时写死了建索引那一刻的 TOP_K；按当前配置覆盖，否则调 TOP_K 时 BM25 仍只取旧值
            bm25_retriever.similarity_top_k = TOP_K
        else:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self._index.docstore,
                similarity_top_k=TOP_K,
            )

        # 消融实验：只用其中一路
        if RETRIEVAL_MODE == "vector":
            self._fusion_retriever = vector_retriever
            return
        if RETRIEVAL_MODE == "bm25":
            self._fusion_retriever = bm25_retriever
            return
        if RETRIEVAL_MODE != "hybrid":
            raise ValueError(f"RETRIEVAL_MODE must be hybrid, vector or bm25, got {RETRIEVAL_MODE!r}")

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

    def retrieve_ranked(self, question: str) -> list[NodeWithScore]:
        """
        Hybrid Top-K → Rerank，返回重排后的完整 Top-K 排序（评测用这个算 Hit@5 / MRR@10）。
        """
        if not self.is_ready:
            raise RuntimeError("知识库未初始化，请先运行: python data_init.py")

        assert self._fusion_retriever is not None
        # Step 1: 混合检索召回 Top-K（内部已完成 RRF 融合）
        candidates: list[NodeWithScore] = self._fusion_retriever.retrieve(question)
        # Step 2: Cross-Encoder / Mock 对全部候选重新排序
        return self._reranker.postprocess_nodes(candidates, query_str=question)

    def retrieve_and_rerank(self, question: str) -> list[NodeWithScore]:
        """
        同步检索链路（CPU/IO 密集，在 FastAPI 中会通过 run_in_executor 异步化）：
        Hybrid Top-K → Rerank → 取前 RERANK_TOP_N 个送给 LLM
        """
        return self.retrieve_ranked(question)[:RERANK_TOP_N]

    def _build_chat_messages(
        self, question: str, nodes: list[NodeWithScore]
    ) -> list[ChatMessage]:
        context = "\n\n---\n\n".join(
            f"[Source {i}: {source_label(n)}]\n{n.get_content()}"
            for i, n in enumerate(nodes, start=1)
        )
        return [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Context excerpts:\n{context}\n\nQuestion:\n{question}",
            ),
        ]

    def answer(self, question: str, nodes: list[NodeWithScore]) -> AnswerResult:
        """非流式生成（评测用），返回用量和模型版本。没有 API Key 时返回 mock 摘录。"""
        start = time.perf_counter()
        if not self.uses_real_llm:
            return AnswerResult(
                text=self._mock_generate(question, nodes),
                model="mock",
                latency_s=time.perf_counter() - start,
            )
        response = Settings.llm.chat(self._build_chat_messages(question, nodes))
        raw = response.raw
        usage = _get(raw, "usage") or {}
        return AnswerResult(
            text=response.message.content or "",
            model=_get(raw, "model") or Settings.llm.model,
            prompt_tokens=int(_get(usage, "prompt_tokens") or 0),
            completion_tokens=int(_get(usage, "completion_tokens") or 0),
            latency_s=time.perf_counter() - start,
            raw_usage=usage if isinstance(usage, dict) else getattr(usage, "model_dump", dict)(),
        )

    @property
    def uses_real_llm(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY", "").strip()) and not isinstance(Settings.llm, MockLLM)

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
                    "title": _source_heading(n),
                    "file_name": n.metadata.get("file_name", ""),
                    "ticker": n.metadata.get("ticker"),
                    "company": n.metadata.get("company"),
                    "fiscal_year": n.metadata.get("fiscal_year"),
                    "page": n.metadata.get("page"),
                    "source_url": n.metadata.get("source_url"),
                    "score": round(float(n.score or 0.0), 4),
                    "preview": _source_preview(n),
                }
                for n in nodes
            ],
        }
        yield f"data: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

        # ── Phase 3: LLM 流式 token 推送（DeepSeek 等走 Chat Completions API）──
        if self.uses_real_llm:
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
        """无 API Key 时，从最相关片段摘录原文，不冒充模型生成。"""
        if not nodes:
            return "演示文档中没有找到相关片段。"
        top = nodes[0]
        section = _source_heading(top)
        lines = [
            re.sub(r"^\s*(?:[-*]\s+|\d+\.\s*)", "", line).replace("**", "").strip()
            for line in top.get_content().splitlines()
            if line.strip() and not line.lstrip().startswith(("#", ">"))
        ]
        excerpt = max(lines, key=lambda line: _phrase_overlap_score(question, line)) if lines else top.get_content().strip()
        return (
            f"演示文档「{section}」中的相关原文：{excerpt} "
            f"（模拟摘录，非 AI 生成；不代表真实法规。）"
        )
