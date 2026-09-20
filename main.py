"""
FastAPI 异步入口 — 金融知识库 RAG 问答服务
==========================================
核心接口：POST /api/chat
- 使用 StreamingResponse + SSE 实现流式输出，降低 TTFT（首字延迟）
- 检索链路在 rag_pipeline 中完成，本层只负责 HTTP 协议转换

SSE（Server-Sent Events）原理：
- 单向推送：服务端 → 客户端（适合 LLM 流式生成场景）
- 基于 HTTP 长连接，Content-Type: text/event-stream
- 每条消息格式：`data: {payload}\n\n`（双换行分隔）
- 对比 WebSocket：SSE 更轻量，天然支持 HTTP/2，FastAPI 原生友好
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline

load_dotenv()

app = FastAPI(
    title="示例文档 RAG 问答系统",
    description="Hybrid Search + RRF + Rerank + SSE 流式输出",
    version="1.0.0",
)

cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

pipeline = RAGPipeline()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户提问", examples=["示例文档中的理赔申请期限是多少？"])


@app.get("/", include_in_schema=False)
async def demo():
    return FileResponse(Path(__file__).parent / "demo" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "kb_ready": pipeline.is_ready}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    SSE 流式问答接口。

    响应事件序列：
    1. data: {"type":"sources", "data":[...]}  — 检索引用（Rerank Top-3）
    2. data: {"type":"token",  "data":"你"}     — LLM 逐 token 推送
    3. data: [DONE]                            — 结束信号
    """
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=503,
            detail="知识库未就绪，请先运行: python data_init.py",
        )

    return StreamingResponse(
        pipeline.stream_chat(req.question),
        media_type="text/event-stream",
        headers={
            # 禁用缓冲，确保 nginx / 代理层也逐条推送
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
