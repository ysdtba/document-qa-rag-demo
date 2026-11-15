import pytest

from eval import metrics
from eval.judge import label_from_points, parse_text_verdict
from eval.run_eval import summarise


def _point(text, covered, contradicted=False):
    return {"point": text, "covered": covered, "contradicted": contradicted}


def test_label_follows_coverage_rules():
    assert label_from_points([_point("a", True), _point("b", True)]) == "correct"
    assert label_from_points([_point("a", True), _point("b", False)]) == "partially_correct"
    assert label_from_points([_point("a", False), _point("b", False)]) == "incorrect"
    assert label_from_points([_point("a", True, contradicted=True)]) == "incorrect"
    assert label_from_points([_point("a", True)], refused=True) == "incorrect"


def test_parse_text_verdict_uses_points_not_the_model_label():
    verdict = parse_text_verdict({
        "points": [_point("pays out 75%", True), _point("reviews surplus capital", False)],
        "refused": False,
        "label": "correct",  # 模型自己标对了，但第二个要点没覆盖
        "reason": "states the payout",
    })
    assert verdict["label"] == "partially_correct"
    assert verdict["model_label"] == "correct"
    assert [point["point"] for point in verdict["points"]] == ["pays out 75%", "reviews surplus capital"]


def test_parse_text_verdict_refusal_and_bad_payloads():
    refused = parse_text_verdict(
        {"points": [_point("a", True)], "refused": False, "reason": ""},
        answer="Not found in the provided annual report excerpts.",
    )
    assert refused["label"] == "incorrect" and refused["refused"]

    with pytest.raises(ValueError):
        parse_text_verdict({"label": "correct", "reason": "no points"})
    with pytest.raises(ValueError):
        parse_text_verdict({"points": [{"point": "a", "covered": "yes"}]})


def test_text_score_counts_partial_as_half_and_parse_error_as_zero():
    assert metrics.text_score(["correct", "correct", "partially_correct", "incorrect"]) == pytest.approx(0.625)
    assert metrics.text_score(["correct", "parse_error"]) == pytest.approx(0.5)
    assert metrics.text_score([]) is None


def test_summarise_reports_text_score():
    records = [
        {"id": "n", "split": "dev", "answer_type": "number", "answer": "1", "correct": True,
         "hit_at_5": True, "rr_at_10": 1.0, "citation_precision": 1.0, "latency_s": 0.1,
         "refused": False, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0},
        {"id": "t1", "split": "dev", "answer_type": "text", "answer": "yes", "text_label": "correct",
         "hit_at_5": True, "rr_at_10": 1.0, "citation_precision": 1.0, "latency_s": 0.2,
         "refused": False, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0},
        {"id": "t2", "split": "dev", "answer_type": "text", "answer": "part", "text_label": "partially_correct",
         "hit_at_5": True, "rr_at_10": 1.0, "citation_precision": 1.0, "latency_s": 0.3,
         "refused": False, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0},
        {"id": "t3", "split": "dev", "answer_type": "text", "answer": "??", "text_label": "parse_error",
         "hit_at_5": False, "rr_at_10": 0.0, "citation_precision": 0.0, "latency_s": 0.4,
         "refused": False, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0},
    ]
    summary = summarise(records)
    assert summary["text_correct"] == pytest.approx(1 / 3)
    assert summary["text_partial"] == pytest.approx(1 / 3)
    assert summary["text_score"] == pytest.approx(0.5)  # (1 + 0.5 + 0) / 3
    assert summary["judge_parse_errors"] == 1
