import json

import pytest

from eval import metrics
from eval.run_eval import summarise


def q(**overrides):
    base = {
        "id": "q1", "ticker": "0388", "fiscal_year": 2025, "question": "Revenue?",
        "answer": "22,374", "answer_type": "number", "unit": "million", "currency": "HKD",
        "gold_pages": [12, 13], "split": "dev", "notes": "",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("answer", [
    "Revenue was HK$22,374 million (HKEX FY2025 p.12).",
    "Revenue was HK$22.374bn.",
    "HK$22,374m",
    "22,400 million HKD",          # within 0.5%
    "In 2025 revenue reached HK$22.37 billion, up 9%.",
])
def test_number_matches_across_units(answer):
    assert metrics.score_number(q(), answer)["correct"]


@pytest.mark.parametrize("answer", [
    "Revenue was HK$22,600 million.",   # 1% off
    "Revenue was US$22,374 million.",   # wrong currency, no FX conversion
    "Revenue was HK$22,374.",           # wrong scale
    "Revenue grew 22,374%.",            # percent, not an amount
    "Not found in the provided annual report excerpts.",
])
def test_number_rejects_wrong_values(answer):
    assert not metrics.score_number(q(), answer)["correct"]


def test_number_without_currency_in_answer_still_matches():
    assert metrics.score_number(q(), "Revenue: 22,374 million")["correct"]


def test_percent_and_per_share_answers():
    assert metrics.score_number(q(answer="9.5", unit="%", currency=""), "Growth was 9.5%.")["correct"]
    assert metrics.score_number(q(answer="9.03", unit="", currency="HKD"), "Basic EPS was HK$9.03.")["correct"]


def test_chinese_scale_words():
    assert metrics.score_number(q(answer="3.2", unit="billion", currency="RMB"), "收入为人民币32亿元")["correct"]


def test_negative_values():
    assert metrics.score_number(q(answer="-1,200", unit="million"), "Change: -1,200 million")["correct"]
    # a year range is not a negative number
    assert all(x.value > 0 for x in metrics.parse_quantities("2024-2025"))


def test_refusal_detection():
    assert metrics.is_refusal("Not found in the provided annual report excerpts.")
    assert not metrics.is_refusal("Revenue was HK$22,374 million.")


def test_retrieval_metrics_need_same_report_and_gold_page():
    question = q()
    chunks = [
        {"ticker": "1299", "fiscal_year": 2025, "page": 12},   # other company, same page
        {"ticker": "0388", "fiscal_year": 2024, "page": 12},   # other year
        {"ticker": "0388", "fiscal_year": 2025, "page": 40},
        {"ticker": "0388", "fiscal_year": 2025, "page": 13},   # first gold hit at rank 4
    ]
    assert metrics.hit_at_k(chunks, question, 5)
    assert not metrics.hit_at_k(chunks, question, 3)
    assert metrics.reciprocal_rank(chunks, question, 10) == pytest.approx(0.25)
    assert metrics.citation_page_precision(chunks[:4], question) == pytest.approx(0.25)


def test_percentile():
    assert metrics.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert metrics.percentile([5], 95) == 5
    assert metrics.percentile([], 50) is None


def test_load_questions_validates(tmp_path):
    good = q()
    bad = q(id="q2", answer_type="number", answer="n/a", gold_pages=[])
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        metrics.load_questions(path)
    message = str(exc.value)
    assert "q2: answerable questions need at least one gold page" in message
    assert "q2: number answer" in message

    path.write_text(json.dumps(good) + "\n" + json.dumps(q(id="q3", answer_type="not_found", answer="", gold_pages=[])) + "\n")
    assert [row["id"] for row in metrics.load_questions(path)] == ["q1", "q3"]


def test_summarise_retrieval_only():
    records = [
        {"id": "a", "split": "dev", "answer_type": "number", "hit_at_5": True, "rr_at_10": 1.0,
         "citation_precision": 1 / 3, "latency_s": 0.2},
        {"id": "b", "split": "dev", "answer_type": "text", "hit_at_5": False, "rr_at_10": 0.0,
         "citation_precision": 0.0, "latency_s": 0.4},
        {"id": "c", "split": "dev", "answer_type": "not_found", "latency_s": 0.3},
    ]
    s = summarise(records)
    assert s["n"] == 3 and s["n_answerable"] == 2
    assert s["hit_at_5"] == pytest.approx(0.5)
    assert s["mrr_at_10"] == pytest.approx(0.5)
    assert "number_accuracy" not in s
