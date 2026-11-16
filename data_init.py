"""
知识库初始化脚本
================
职责：
1. 扫描 DOCS_DIR（默认 docs/；年报用 DOCS_DIR=data/raw）下的 .md / .txt / .pdf；
   PDF 按页抽文本，每页带 ticker / company / fiscal_year / page（来自 data/manifest.csv）；
2. 使用 Semantic / Markdown 语义切块（而非固定字数硬切）；
3. 写入 Chroma 向量库 + Docstore + BM25 索引，供 RAG 管道加载。

运行：python data_init.py
添加文档：直接把新文件丢进 docs/ 文件夹，重新运行即可重建全量索引。
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv()

# ── 路径配置 ──────────────────────────────────────────────────────────
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
MANIFEST_PATH = Path(os.getenv("MANIFEST_PATH", "data/manifest.csv"))
CHROMA_PATH = os.getenv("CHROMA_PATH", "./storage/chroma")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "./storage/docstore")
BM25_PATH = os.getenv("BM25_PATH", "./storage/bm25")

# 支持的文件后缀 → reader 映射
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

# 只让这些 metadata 进入 embedding 和 LLM 上下文；其余字段（网址、哈希等）只用于引用和评测，
# 也避免它们挤占 chunk_size。
VISIBLE_METADATA_KEYS = {"company", "ticker", "fiscal_year", "page", "section"}


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, str]]:
    """读取 data/manifest.csv，按 file_name 索引。没有 manifest 时返回空字典。"""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["file_name"]: row for row in csv.DictReader(fh)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pdf_pages(path: Path) -> list[str]:
    """按页读取 PDF 文本（需安装 pypdf）。返回列表的下标 i 对应 PDF 第 i+1 页。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            f"PDF 文件需要 pypdf 库，请运行: pip install pypdf\n"
            f"（文件: {path}）"
        )
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def _manifest_metadata(row: dict[str, str]) -> dict:
    meta = {
        "ticker": row.get("ticker", ""),
        "company": row.get("company", ""),
        "language": row.get("language", ""),
        "source_url": row.get("source_url", ""),
    }
    if row.get("fiscal_year", "").isdigit():
        meta["fiscal_year"] = int(row["fiscal_year"])
    return {k: v for k, v in meta.items() if v != ""}


def _make_document(text: str, metadata: dict) -> Document:
    hidden = sorted(set(metadata) - VISIBLE_METADATA_KEYS)
    return Document(
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=hidden,
        excluded_llm_metadata_keys=hidden,
    )


def _tables_enabled() -> bool:
    from table_chunks import table_chunks_enabled
    return table_chunks_enabled()


def _page_words(path: Path) -> list[list[dict]]:
    from table_chunks import read_document_words
    return read_document_words(path)


def _documents_for_page(text: str, base_meta: dict, label: str, page_no: int, words: list[dict] | None) -> list[Document]:
    """One page. TABLE_CHUNKS=off keeps the pypdf text and adds no chunk_type."""
    meta = {**base_meta, "page": page_no, "section": f"{label} p.{page_no}"}
    if words is None:
        return [_make_document(text, meta)] if text.strip() else []

    from table_chunks import detect_tables, prose_without_tables, table_to_chunks

    tables = detect_tables(words)
    documents: list[Document] = []
    if not tables:
        if text.strip():
            documents.append(_make_document(text, {**meta, "chunk_type": "text"}))
        return documents

    prose = prose_without_tables(words, tables)
    if prose.strip():
        documents.append(_make_document(prose, {**meta, "chunk_type": "text"}))
    context = f"{meta.get('company', label)} FY{meta.get('fiscal_year', '')} p.{page_no}".strip()
    for table in tables:
        for chunk in table_to_chunks(table, context=context):
            documents.append(_make_document(chunk, {**meta, "chunk_type": "table"}))
    return documents


def load_documents(docs_dir: Path, manifest: dict[str, dict[str, str]] | None = None) -> list[Document]:
    """
    扫描目录，加载所有支持的文档。

    支持格式：
      - .md  / .txt  → 直接按 UTF-8 文本读取，一个文件一个 Document
      - .pdf         → 通过 pypdf 按页提取，一页一个 Document，metadata 带 page（PDF 页码，从 1 开始）
    PDF 在 manifest 里有记录时，附加 ticker / company / fiscal_year 等字段，并核对 SHA256。
    """
    manifest = load_manifest() if manifest is None else manifest
    documents: list[Document] = []
    files = sorted(
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not files:
        raise FileNotFoundError(
            f"docs/ 目录下未找到支持的文档（{' / '.join(sorted(SUPPORTED_SUFFIXES))}）。"
            f"请将文档放入 {docs_dir.resolve()} 后重新运行。"
        )

    print(f"[扫描] 在 {docs_dir.resolve()} 中发现 {len(files)} 个文档:")
    for f in files:
        print(f"        - {f.name}")

    for filepath in files:
        suffix = filepath.suffix.lower()
        base_meta = {
            "source": str(filepath),
            "file_name": filepath.name,
            "file_type": suffix.lstrip("."),
        }

        if suffix == ".pdf":
            row = manifest.get(filepath.name)
            if row is None:
                print(f"  ⚠ {filepath.name} 不在 manifest 中，缺少 ticker/财年；请运行 python scripts/build_manifest.py")
            else:
                base_meta.update(_manifest_metadata(row))
                if row.get("sha256") and row["sha256"] != _sha256(filepath):
                    print(f"  ⚠ {filepath.name} 的 SHA256 与 manifest 不一致，请核对文件来源")
            label = (
                f"{base_meta['company']} FY{base_meta['fiscal_year']}"
                if base_meta.get("company") and base_meta.get("fiscal_year")
                else filepath.name
            )
            pages = _read_pdf_pages(filepath)
            page_words = _page_words(filepath) if _tables_enabled() else None
            kept = 0
            for page_no, text in enumerate(pages, start=1):
                page_docs = _documents_for_page(
                    text,
                    base_meta,
                    label,
                    page_no,
                    None if page_words is None else page_words[page_no - 1],
                )
                documents.extend(page_docs)
                kept += len(page_docs)
            print(f"        {filepath.name}: {len(pages)} 页，{kept} 个页面文档")
            if kept == 0:
                print(f"  ⚠ {filepath.name} 没有可抽取的文本（可能是扫描件）")
            continue

        text = filepath.read_text(encoding="utf-8")
        if not text.strip():
            print(f"  ⚠ 跳过空文件: {filepath.name}")
            continue
        documents.append(_make_document(text, base_meta))

    if not documents:
        raise RuntimeError("docs/ 中所有文件均为空，无法初始化索引。")

    return documents


def _node_from_document(document: Document) -> TextNode:
    return TextNode(
        text=document.text,
        metadata=dict(document.metadata),
        excluded_embed_metadata_keys=list(document.excluded_embed_metadata_keys),
        excluded_llm_metadata_keys=list(document.excluded_llm_metadata_keys),
    )


def _split_text_documents(documents: list[Document]) -> list:
    """Markdown 标题切分后再按句切。表格 chunk 不走这里，避免表头和数字再次分开。"""
    md_parser = MarkdownNodeParser()
    md_nodes = md_parser.get_nodes_from_documents(documents)
    splitter = SentenceSplitter(chunk_size=384, chunk_overlap=48)
    final_nodes = splitter.get_nodes_from_documents(
        [
            Document(
                text=n.get_content(),
                metadata=n.metadata,
                excluded_embed_metadata_keys=n.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=n.excluded_llm_metadata_keys,
            )
            for n in md_nodes
        ]
    )
    for node in final_nodes:
        headers = (node.metadata.get("header_path", "") or node.metadata.get("Header Path", "")).strip(" /")
        source_file = node.metadata.get("file_name", "")
        if "page" in node.metadata:
            node.metadata.pop("header_path", None)  # PDF 文本没有 Markdown 标题，保留按页生成的 section
        elif headers:
            node.metadata["section"] = f"{source_file} / {headers}" if source_file else headers
        else:
            node.metadata.setdefault("section", source_file or "未知文档")
    return final_nodes


def semantic_chunk_documents(documents: list[Document]) -> list:
    """
    语义切块策略：
    ─────────────
    1. MarkdownNodeParser：按 # / ## / ### 标题层级切分，保留文档结构语义；
       金融合规文档天然按「章-节-条」组织，按标题切比固定 512 token 更合理。
    2. SentenceSplitter：对过长章节做二次切分（chunk_size=384, overlap=48），
       保证每个 chunk 可独立嵌入且上下文连贯。
    3. chunk_type=table 的文档整块保留，不按句子切开。
    """
    final_nodes = []
    buffer: list[Document] = []

    def flush() -> None:
        if buffer:
            final_nodes.extend(_split_text_documents(buffer))
            buffer.clear()

    for document in documents:
        if document.metadata.get("chunk_type") == "table":
            flush()
            final_nodes.append(_node_from_document(document))
        else:
            buffer.append(document)
    flush()
    return final_nodes


def main() -> None:
    print("=" * 60)
    print("  知识库 RAG — 索引初始化")
    print("=" * 60)

    # 1. 扫描文档
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print(f"[提示] 已创建 {DOCS_DIR}/ 目录，请放入文档后重新运行。")
        return
    documents = load_documents(DOCS_DIR)
    print(f"[1/4] 已加载 {len(documents)} 个文档")

    # 2. 初始化 Embedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
    )
    print(f"[2/4] Embedding 模型: {Settings.embed_model}")

    # 3. 语义切块
    nodes = semantic_chunk_documents(documents)
    if _tables_enabled():
        table_count = sum(1 for node in nodes if node.metadata.get("chunk_type") == "table")
        print(f"[3/4] 语义切块完成: {len(nodes)} 个 chunk（表格 {table_count}，文字 {len(nodes) - table_count}）")
    else:
        print(f"[3/4] 语义切块完成: {len(nodes)} 个 chunk")

    # 4. 写入 Chroma + Docstore + BM25
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    Path(DOCSTORE_PATH).mkdir(parents=True, exist_ok=True)
    Path(BM25_PATH).mkdir(parents=True, exist_ok=True)

    docstore = SimpleDocumentStore()
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection("hkex_finance_kb")
    except Exception:
        pass
    collection = db.create_collection("hkex_finance_kb")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)

    docstore_file = Path(DOCSTORE_PATH) / "docstore.json"
    docstore_file.parent.mkdir(parents=True, exist_ok=True)
    storage.docstore.persist(persist_path=str(docstore_file))
    bm25 = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=int(os.getenv("TOP_K", "20")),
    )
    bm25.persist(str(BM25_PATH))

    print(f"[4/4] 索引持久化完成:")
    print(f"       Chroma  → {CHROMA_PATH}")
    print(f"       Docstore → {DOCSTORE_PATH}")
    print(f"       BM25    → {BM25_PATH}")
    print()
    print("初始化成功！请启动服务: uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
