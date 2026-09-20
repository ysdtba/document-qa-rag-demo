#!/usr/bin/env python3
"""
SSE 流式测试客户端
==================
一键运行，在终端观察「一个字一个字往外蹦」的流式效果。

用法：
    # 先启动服务
    uvicorn main:app --reload --port 8000

    # 另开终端运行本脚本
    python test_client.py
    python test_client.py "OTC 异常理赔的赔偿上限是多少？"
"""

from __future__ import annotations

import json
import sys
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
DEFAULT_QUESTION = "OTC 异常理赔申请须在多少个交易日内提交？赔偿上限是多少？"


def run_chat(question: str) -> None:
    print("=" * 60)
    print(f"  问题: {question}")
    print("=" * 60)

    # stream=True 开启 SSE 长连接，iter_lines 逐行读取
    with httpx.stream(
        "POST",
        f"{BASE_URL}/api/chat",
        json={"question": question},
        timeout=120.0,
    ) as response:
        if response.status_code != 200:
            print(f"[错误] HTTP {response.status_code}: {response.read().decode()}")
            return

        print("\n⏳ 等待 SSE 流式响应...\n")

        sources_printed = False
        answer_header_printed = False
        first_token = True
        t0 = time.perf_counter()

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue

            payload = line[6:]  # 去掉 "data: " 前缀
            if payload == "[DONE]":
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"\n\n{'─' * 40}")
                print(f"✅ 流式完成 | 总耗时 {elapsed:.0f}ms")
                break

            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "sources":
                if not sources_printed:
                    print("📚 检索引用 (Rerank Top-3):")
                    print("-" * 40)
                    sources_printed = True
                for i, src in enumerate(event["data"], 1):
                    print(f"  [{i}] {src['section']}  (score={src['score']})")
                    print(f"      {src['preview']}")
                print()
            elif event.get("type") == "token":
                if not answer_header_printed:
                    print("🤖 回答 (SSE 流式):")
                    print("-" * 40)
                    answer_header_printed = True
                if first_token:
                    ttft = (time.perf_counter() - t0) * 1000
                    print(f"(TTFT ≈ {ttft:.0f}ms)\n")
                    first_token = False
                print(event["data"], end="", flush=True)


def main() -> None:
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUESTION

    # 健康检查
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        health = r.json()
        if not health.get("kb_ready"):
            print("⚠️  知识库未就绪，请先运行: python data_init.py")
            sys.exit(1)
    except httpx.ConnectError:
        print("❌ 无法连接服务，请先启动: uvicorn main:app --reload --port 8000")
        sys.exit(1)

    run_chat(question)


if __name__ == "__main__":
    main()
