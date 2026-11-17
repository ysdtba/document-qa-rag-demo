"""LLM-as-judge for text answers and faithfulness.

Uses any OpenAI-compatible Chat Completions endpoint, configured separately from
the generator so the judge can be a stronger model:

    JUDGE_MODEL                      (required to enable judging)
    JUDGE_API_KEY, JUDGE_BASE_URL    (default to OPENAI_API_KEY / OPENAI_BASE_URL)
    JUDGE_THINKING                   (enabled / disabled; DeepSeek ignores temperature when thinking)
    JUDGE_PRICE_INPUT_PER_M, JUDGE_PRICE_OUTPUT_PER_M   (USD per 1M tokens, for cost tracking)

temperature is fixed at 0 and the model version returned by the API is recorded.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

TEXT_LABELS = {"correct", "partially_correct", "incorrect"}
FAITHFUL_LABELS = {"supported", "partially_supported", "unsupported"}

# 文字题评判提示的版本。改了 TEXT_PROMPT 或标签规则就换一个新值，写进评测结果。
JUDGE_PROMPT_VERSION = "text-v2"
TEXT_MAX_TOKENS = 800  # 要点列表比一句理由长，400 容易被截断

TEXT_PROMPT = """You grade an answer to a question about a company's annual report.

Question: {question}
Reference answer (written and checked by a human): {gold}
Candidate answer: {answer}

Split the reference answer into its distinct factual points. Ignore style and citation format.
For each point, decide whether the candidate states that point:
- covered: the candidate states this point (wording may differ).
- contradicted: the candidate states the opposite of this point. Leaving a point out is not a contradiction.
A refusal such as "Not found in the provided annual report excerpts." covers no point.

Reply with JSON only:
{{
  "points": [{{"point": "<one fact from the reference>", "covered": true, "contradicted": false}}],
  "refused": false,
  "label": "correct" | "partially_correct" | "incorrect",
  "reason": "<one sentence>"
}}

Set label using only these rules:
- correct: every point is covered, none is contradicted, and the candidate does not refuse.
- partially_correct: at least one point is covered and at least one is not, none is contradicted, and the candidate does not refuse.
- incorrect: no point is covered, or any point is contradicted, or the candidate refuses.
"""

FAITHFUL_PROMPT = """You check whether an answer is supported by the source excerpts it was given.

Source excerpts:
{context}

Answer: {answer}

Check every factual claim in the answer (numbers, names, dates, statements) against the excerpts.
Do not use outside knowledge; a claim that is true but absent from the excerpts is unsupported.
- "supported": every claim is backed by the excerpts.
- "partially_supported": at least one claim is not backed.
- "unsupported": the main claim is not backed.

Reply with JSON only: {{"label": "supported" | "partially_supported" | "unsupported", "unsupported_claims": ["..."], "reason": "<one sentence>"}}"""


@dataclass
class JudgeResult:
    label: str
    reason: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: dict = field(default_factory=dict)


def _judge_key() -> str:
    return (os.getenv("JUDGE_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip())


def _judge_base_url() -> str | None:
    return os.getenv("JUDGE_BASE_URL", "").strip() or os.getenv("OPENAI_BASE_URL", "").strip() or None


def _judge_thinking() -> str:
    return os.getenv("JUDGE_THINKING", "").strip().lower()


def judge_configured() -> bool:
    return bool(_judge_key() and os.getenv("JUDGE_MODEL", "").strip())


def judge_config() -> dict:
    return {
        "judge_model": os.getenv("JUDGE_MODEL") if judge_configured() else None,
        "judge_base_url": _judge_base_url() if judge_configured() else None,
        "judge_temperature": 0,
        "judge_thinking": _judge_thinking() or None,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
    }


def label_from_points(points: list[dict], *, refused: bool = False) -> str:
    """标签只由覆盖情况决定，不采用模型自己给的 label。"""
    if refused:
        return "incorrect"
    if not points:
        raise ValueError("text verdict has no points")
    covered = [bool(point["covered"]) for point in points]
    contradicted = any(point.get("contradicted") for point in points)
    if contradicted or not any(covered):
        return "incorrect"
    if all(covered):
        return "correct"
    return "partially_correct"


def parse_text_verdict(data: dict, answer: str = "") -> dict:
    """把评判返回的 JSON 收成要点列表，并按固定规则算出 label。解析不了就抛错。"""
    from eval import metrics

    raw_points = data.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("text verdict needs a non-empty points list")
    points: list[dict] = []
    for item in raw_points:
        if not isinstance(item, dict):
            raise ValueError("each point must be an object")
        text = str(item.get("point", "")).strip()
        if not text or not isinstance(item.get("covered"), bool):
            raise ValueError("each point needs a point string and a boolean covered")
        contradicted = item.get("contradicted", False)
        if not isinstance(contradicted, bool):
            raise ValueError("contradicted must be a boolean")
        points.append({"point": text, "covered": item["covered"], "contradicted": contradicted})
    refused_flag = data.get("refused", False)
    if not isinstance(refused_flag, bool):
        raise ValueError("refused must be a boolean")
    refused = refused_flag or metrics.is_refusal(answer)
    return {
        "points": points,
        "refused": refused,
        "label": label_from_points(points, refused=refused),
        "reason": str(data.get("reason", "")),
        "model_label": str(data.get("label", "")).strip().lower(),
    }


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge reply has no JSON object: {text[:200]!r}")
    return json.loads(match.group(0))


class Judge:
    def __init__(self) -> None:
        if not judge_configured():
            raise RuntimeError("Set JUDGE_MODEL (and an API key) in .env to run LLM judging.")
        from openai import OpenAI

        self.model = os.environ["JUDGE_MODEL"]
        self.client = OpenAI(api_key=_judge_key(), base_url=_judge_base_url())
        thinking = _judge_thinking()
        self.extra_body = {"thinking": {"type": thinking}} if thinking in {"enabled", "disabled"} else None

    def _complete(self, prompt: str, max_tokens: int):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            extra_body=self.extra_body,
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            response.model or self.model,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    def _ask(self, prompt: str, allowed: set[str]) -> JudgeResult:
        content, model, prompt_tokens, completion_tokens = self._complete(prompt, max_tokens=400)
        try:
            data = _parse_json(content)
            label = str(data.get("label", "")).strip().lower()
        except (ValueError, json.JSONDecodeError):
            data, label = {"reason": content[:300]}, "parse_error"
        if label not in allowed:
            label = "parse_error"
        return JudgeResult(
            label=label,
            reason=str(data.get("reason", "")),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            extra={k: v for k, v in data.items() if k not in {"label", "reason"}},
        )

    def grade_text(self, question: str, gold: str, answer: str) -> JudgeResult:
        content, model, prompt_tokens, completion_tokens = self._complete(
            TEXT_PROMPT.format(question=question, gold=gold, answer=answer),
            max_tokens=TEXT_MAX_TOKENS,
        )
        try:
            verdict = parse_text_verdict(_parse_json(content), answer)
        except (ValueError, json.JSONDecodeError):
            return JudgeResult(
                label="parse_error",
                reason=content[:300],
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        extra = {"points": verdict["points"], "refused": verdict["refused"]}
        if verdict["model_label"] and verdict["model_label"] != verdict["label"]:
            extra["model_label"] = verdict["model_label"]
        return JudgeResult(
            label=verdict["label"],
            reason=verdict["reason"],
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            extra=extra,
        )

    def grade_faithfulness(self, context: str, answer: str) -> JudgeResult:
        return self._ask(FAITHFUL_PROMPT.format(context=context, answer=answer), FAITHFUL_LABELS)
