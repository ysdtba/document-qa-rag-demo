"""Deterministic metrics: number parsing/normalisation, retrieval hits, refusal check.

Nothing here calls an LLM, so everything is covered by unit tests.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

NUMBER_TOLERANCE = 0.005  # 0.5% relative

# ── Questions file ────────────────────────────────────────────────────

REQUIRED_FIELDS = {
    "id", "ticker", "fiscal_year", "question", "answer", "answer_type",
    "unit", "currency", "gold_pages", "split", "notes",
}
ANSWER_TYPES = {"number", "text", "not_found"}
SPLITS = {"dev", "test"}


def load_questions(path: Path) -> list[dict]:
    """Read eval/questions.jsonl and validate every row. Never modifies the file."""
    questions: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"line {line_no}: missing fields {sorted(missing)}")
            continue
        qid = str(row["id"])
        if qid in seen:
            errors.append(f"line {line_no}: duplicate id {qid}")
        seen.add(qid)
        if row["answer_type"] not in ANSWER_TYPES:
            errors.append(f"{qid}: answer_type must be one of {sorted(ANSWER_TYPES)}")
        if row["split"] not in SPLITS:
            errors.append(f"{qid}: split must be dev or test")
        if not isinstance(row["gold_pages"], list) or not all(isinstance(p, int) for p in row["gold_pages"]):
            errors.append(f"{qid}: gold_pages must be a list of PDF page integers")
        elif row["answer_type"] != "not_found" and not row["gold_pages"]:
            errors.append(f"{qid}: answerable questions need at least one gold page")
        if row["answer_type"] == "number" and not parse_quantities(f"{row['answer']}"):
            errors.append(f"{qid}: number answer {row['answer']!r} has no parseable number")
        questions.append(row)
    if errors:
        raise ValueError("questions.jsonl has problems:\n  " + "\n  ".join(errors))
    return questions


# ── Numbers ───────────────────────────────────────────────────────────

SCALE_WORDS = {
    "trillion": 1e12, "tn": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9, "十亿": 1e9,
    "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6, "百万": 1e6,
    "thousand": 1e3, "k": 1e3, "'000": 1e3, "千": 1e3,
    "亿": 1e8, "万": 1e4,
}

CURRENCY_ALIASES = {
    "HKD": ["hk$", "hkd", "港元", "港币", "港幣"],
    "USD": ["us$", "usd", "美元"],
    "RMB": ["rmb", "cny", "人民币", "人民幣", "¥", "￥"],
}

_SCALE_ALT = "|".join(sorted((re.escape(w) for w in SCALE_WORDS), key=len, reverse=True))
_CUR_ALT = "|".join(sorted((re.escape(a) for v in CURRENCY_ALIASES.values() for a in v), key=len, reverse=True))

# currency? sign? number scale? currency? %?
QUANTITY_RE = re.compile(
    rf"(?P<cur1>{_CUR_ALT})?\s*"
    rf"(?P<neg>(?<![\w.])[-−])?\s*"
    rf"(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    rf"\s*(?P<scale>(?:{_SCALE_ALT})(?![a-z]))?"
    rf"\s*(?P<cur2>{_CUR_ALT})?"
    rf"\s*(?P<pct>%|percent|per cent)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Quantity:
    value: float          # already multiplied by scale
    currency: str | None  # HKD / USD / RMB / None
    is_percent: bool


def normalise_currency(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().lower()
    for code, aliases in CURRENCY_ALIASES.items():
        if t == code.lower() or t in aliases:
            return code
    return None


def scale_factor(unit: str | None) -> float:
    """Gold-answer unit field → multiplier ('million' → 1e6, '' or 'HK$' → 1)."""
    if not unit:
        return 1.0
    return SCALE_WORDS.get(unit.strip().lower(), 1.0)


def parse_quantities(text: str) -> list[Quantity]:
    """Find every number in text with its scale, currency and % marker."""
    found: list[Quantity] = []
    for m in QUANTITY_RE.finditer(text):
        raw = m.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        scale = m.group("scale")  # the (?![a-z]) guard stops "5 members" reading as 5 million
        if scale:
            value *= SCALE_WORDS[scale.lower()]
        if m.group("neg"):
            value = -value
        found.append(Quantity(
            value=value,
            currency=normalise_currency(m.group("cur1") or m.group("cur2")),
            is_percent=bool(m.group("pct")),
        ))
    return found


def gold_quantity(question: dict) -> Quantity:
    """The expected value of a number question, using its unit and currency fields."""
    unit = (question.get("unit") or "").strip()
    parsed = parse_quantities(str(question["answer"]))
    if not parsed:
        raise ValueError(f"{question['id']}: cannot parse gold answer {question['answer']!r}")
    q = parsed[0]
    is_percent = q.is_percent or unit in {"%", "percent"}
    value = q.value * scale_factor(unit)
    currency = normalise_currency(question.get("currency")) or q.currency
    return Quantity(value=value, currency=currency, is_percent=is_percent)


def numbers_match(expected: float, actual: float, tolerance: float = NUMBER_TOLERANCE) -> bool:
    if expected == 0:
        return abs(actual) <= tolerance
    return math.isclose(actual, expected, rel_tol=tolerance)


def score_number(question: dict, answer_text: str) -> dict:
    """
    Correct when any number in the answer equals the gold value within 0.5% after
    scaling (million/billion/…) and its currency, if stated, matches the gold currency.
    No FX conversion: HK$ vs US$ figures are different answers.
    """
    gold = gold_quantity(question)
    candidates = parse_quantities(answer_text)
    for cand in candidates:
        if gold.is_percent != cand.is_percent and cand.is_percent:
            continue
        if gold.currency and cand.currency and gold.currency != cand.currency:
            continue
        if numbers_match(gold.value, cand.value):
            return {"correct": True, "gold_value": gold.value, "matched_value": cand.value}
    return {
        "correct": False,
        "gold_value": gold.value,
        "candidates": [c.value for c in candidates[:10]],
    }


# ── Refusal ───────────────────────────────────────────────────────────

REFUSAL_PATTERNS = [
    re.compile(r"not found in the provided annual report excerpts", re.IGNORECASE),
]


def is_refusal(answer_text: str) -> bool:
    return any(p.search(answer_text) for p in REFUSAL_PATTERNS)


# ── Retrieval ─────────────────────────────────────────────────────────


def chunk_is_gold(chunk: dict, question: dict) -> bool:
    """A retrieved chunk counts when it comes from the same report and a gold page."""
    if str(chunk.get("ticker")) != str(question["ticker"]):
        return False
    if chunk.get("fiscal_year") is not None and int(chunk["fiscal_year"]) != int(question["fiscal_year"]):
        return False
    return chunk.get("page") in set(question["gold_pages"])


def hit_at_k(chunks: list[dict], question: dict, k: int = 5) -> bool:
    return any(chunk_is_gold(c, question) for c in chunks[:k])


def reciprocal_rank(chunks: list[dict], question: dict, k: int = 10) -> float:
    for rank, chunk in enumerate(chunks[:k], start=1):
        if chunk_is_gold(chunk, question):
            return 1.0 / rank
    return 0.0


def citation_page_precision(chunks: list[dict], question: dict) -> float | None:
    """Share of the chunks sent to the LLM that sit on a gold page."""
    if not chunks:
        return None
    return sum(chunk_is_gold(c, question) for c in chunks) / len(chunks)


# ── Aggregation ───────────────────────────────────────────────────────


def percentile(values: Iterable[float], pct: float) -> float | None:
    data = sorted(values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * pct / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def mean(values: Iterable[float | None]) -> float | None:
    data = [v for v in values if v is not None]
    return sum(data) / len(data) if data else None
