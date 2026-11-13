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

TEXT_PROMPT = """You grade an answer to a question about a company's annual report.

Question: {question}
Reference answer (written and checked by a human): {gold}
Candidate answer: {answer}

Grade the candidate against the reference only. Ignore style and citation format.
- "correct": states the reference's key facts with no contradicting claims.
- "partially_correct": some key facts right, others missing or wrong.
- "incorrect": wrong, contradicts the reference, or refuses to answer.

Reply with JSON only: {{"label": "correct" | "partially_correct" | "incorrect", "reason": "<one sentence>"}}"""

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

    def _ask(self, prompt: str, allowed: set[str]) -> JudgeResult:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
            extra_body=self.extra_body,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
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
            model=response.model or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            extra={k: v for k, v in data.items() if k not in {"label", "reason"}},
        )

    def grade_text(self, question: str, gold: str, answer: str) -> JudgeResult:
        return self._ask(TEXT_PROMPT.format(question=question, gold=gold, answer=answer), TEXT_LABELS)

    def grade_faithfulness(self, context: str, answer: str) -> JudgeResult:
        return self._ask(FAITHFUL_PROMPT.format(context=context, answer=answer), FAITHFUL_LABELS)
