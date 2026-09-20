"""Smoke-test the running, key-free sample demo using only the standard library."""

import json
import os
from urllib import request


BASE_URL = os.environ.get("DEMO_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
QUESTION = "示例文档中的理赔申请期限是多少？"


def main():
    with request.urlopen(f"{BASE_URL}/health", timeout=10) as response:
        health = json.load(response)
    assert health == {"status": "ok", "kb_ready": True}, health

    payload = json.dumps({"question": QUESTION}, ensure_ascii=False).encode("utf-8")
    chat_request = request.Request(
        f"{BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    with request.urlopen(chat_request, timeout=60) as response:
        assert response.headers.get_content_type() == "text/event-stream"
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("data: "):
                events.append(line[6:])

    assert events and events[-1] == "[DONE]", events[-1:]
    messages = [json.loads(event) for event in events[:-1]]
    sources = next(message["data"] for message in messages if message["type"] == "sources")
    answer = "".join(message["data"] for message in messages if message["type"] == "token")
    assert sources and "理赔申请流程" in sources[0]["title"], sources
    assert "5 个交易日" in answer, answer
    assert "模拟摘录，非 AI 生成" in answer, answer
    print("PASS: health ready; SSE complete; relevant citation; 5-trading-day mock answer")


if __name__ == "__main__":
    main()
