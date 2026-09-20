"""
知识库初始化脚本
================
职责：
1. 扫描 docs/ 目录下所有支持的文档（.md / .txt / .pdf）；
2. 使用 Semantic / Markdown 语义切块（而非固定字数硬切）；
3. 写入 Chroma 向量库 + Docstore + BM25 索引，供 RAG 管道加载。

运行：python data_init.py
添加文档：直接把新文件丢进 docs/ 文件夹，重新运行即可重建全量索引。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv()

# ── 路径配置 ──────────────────────────────────────────────────────────
DOCS_DIR = Path("docs")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./storage/chroma")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "./storage/docstore")
BM25_PATH = os.getenv("BM25_PATH", "./storage/bm25")

# 支持的文件后缀 → reader 映射
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _read_pdf(path: Path) -> str:
    """读取 PDF 文本内容（需安装 pypdf）。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            f"PDF 文件需要 pypdf 库，请运行: pip install pypdf\n"
            f"（文件: {path}）"
        )
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(docs_dir: Path) -> list[Document]:
    """
    扫描 docs/ 目录，加载所有支持的文档。

    支持格式：
      - .md  / .txt  → 直接按 UTF-8 文本读取
      - .pdf         → 通过 pypdf 提取文本
    """
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
        if suffix == ".pdf":
            text = _read_pdf(filepath)
        else:
            text = filepath.read_text(encoding="utf-8")

        if not text.strip():
            print(f"  ⚠ 跳过空文件: {filepath.name}")
            continue

        documents.append(Document(
            text=text,
            metadata={
                "source": str(filepath),
                "file_name": filepath.name,
                "file_type": suffix.lstrip("."),
            },
        ))

    if not documents:
        raise RuntimeError("docs/ 中所有文件均为空，无法初始化索引。")

    return documents


def semantic_chunk_documents(documents: list[Document]) -> list:
    """
    语义切块策略：
    ─────────────
    1. MarkdownNodeParser：按 # / ## / ### 标题层级切分，保留文档结构语义；
       金融合规文档天然按「章-节-条」组织，按标题切比固定 512 token 更合理。
    2. SentenceSplitter：对过长章节做二次切分（chunk_size=384, overlap=48），
       保证每个 chunk 可独立嵌入且上下文连贯。
    3. 非 Markdown 文件（.txt / .pdf）也能被 SentenceSplitter 正常切分。
    """
    md_parser = MarkdownNodeParser()
    md_nodes = md_parser.get_nodes_from_documents(documents)

    splitter = SentenceSplitter(chunk_size=384, chunk_overlap=48)
    final_nodes = splitter.get_nodes_from_documents(
        [Document(text=n.get_content(), metadata=n.metadata) for n in md_nodes]
    )

    # 注入 section 元数据，便于 RAG 回答时引用来源
    for node in final_nodes:
        headers = (node.metadata.get("header_path", "") or node.metadata.get("Header Path", "")).strip(" /")
        source_file = node.metadata.get("file_name", "")
        if headers:
            node.metadata["section"] = f"{source_file} / {headers}" if source_file else headers
        else:
            node.metadata.setdefault("section", source_file or "未知文档")

    return final_nodes


def main() -> None:
    print("=" * 60)
    print("  知识库 RAG — 索引初始化")
    print("=" * 60)

    # 1. 扫描文档
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir()
        print(f"[提示] 已创建 docs/ 目录，请放入文档后重新运行。")
        return
    documents = load_documents(DOCS_DIR)
    print(f"[1/4] 已加载 {len(documents)} 个文档")

    # 2. 初始化 Embedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
    )
    print(f"[2/4] Embedding 模型: {Settings.embed_model}")

    # 3. 语义切块
    nodes = semantic_chunk_documents(documents)
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
        similarity_top_k=int(os.getenv("TOP_K", "10")),
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
