"""Regrade text answers in an existing result file with the current judge prompt.

Does not retrieve or generate. Reads the saved answers and the gold answers in
eval/questions.jsonl (never modified).

    python eval/rejudge.py eval/results/2026-11-13_embed_en.json --repeat 3

Writes eval/results/<today>_<original run>_rejudge.json and regenerates RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from eval import metrics  # noqa: E402
from eval.judge import JUDGE_PROMPT_VERSION, Judge, judge_config, judge_configured  # noqa: E402
from eval.run_eval import (  # noqa: E402
    QUESTIONS_PATH,
    RESULTS_DIR,
    _cost,
    _git_state,
    _sha256,
    summarise,
    write_results_md,
)


def rejudge(source: dict, questions: dict[str, dict], judge: Judge, repeat: int) -> tuple[list[dict], dict]:
    """Return new records and a small report of how often the repeated labels agreed."""
    records = []
    judged = consistent = 0
    prompt_tokens = completion_tokens = 0
    for record in source["records"]:
        record = dict(record)
        if record.get("answer_type") == "text" and "answer" in record:
            gold = questions.get(record["id"])
            if gold is None:
                raise KeyError(f"{record['id']} is not in the questions file")
            verdicts = [
                judge.grade_text(record["question"], str(gold["answer"]), record["answer"])
                for _ in range(repeat)
            ]
            labels = [verdict.label for verdict in verdicts]
            first = verdicts[0]
            record["text_label_previous"] = record.get("text_label")
            record["text_label"] = first.label
            record["text_reason"] = first.reason
            record["text_points"] = first.extra.get("points", [])
            record["text_label_repeats"] = labels
            record["text_label_consistent"] = len(set(labels)) == 1
            record["correct"] = first.label == "correct"
            record["judge_model"] = first.model
            cost = sum(_cost(v.prompt_tokens, v.completion_tokens, "JUDGE") for v in verdicts)
            record["judge_cost_usd"] = record.get("judge_cost_usd", 0.0) + cost
            prompt_tokens += sum(v.prompt_tokens for v in verdicts)
            completion_tokens += sum(v.completion_tokens for v in verdicts)
            judged += 1
            consistent += record["text_label_consistent"]
            print(f"{record['id']}: {record.get('text_label_previous')} -> {', '.join(labels)}")
        records.append(record)
    report = {
        "prompt_version": JUDGE_PROMPT_VERSION,
        "repeat": repeat,
        "text_judged": judged,
        "text_consistent": consistent,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": _cost(prompt_tokens, completion_tokens, "JUDGE"),
    }
    return records, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", type=Path, help="existing eval/results/*.json")
    parser.add_argument("--repeat", type=int, default=1, help="grade each text answer this many times")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    args = parser.parse_args()
    if args.repeat < 1:
        print("--repeat must be at least 1")
        return 1
    if not judge_configured():
        print("Set JUDGE_MODEL and an API key in .env.")
        return 1

    source = json.loads(args.result.read_text(encoding="utf-8"))
    questions = {row["id"]: row for row in metrics.load_questions(args.questions)}
    records, report = rejudge(source, questions, Judge(), args.repeat)

    splits = sorted({r["split"] for r in records})
    summary = {split: summarise([r for r in records if r["split"] == split]) for split in splits}
    if len(splits) > 1:
        summary["all"] = summarise(records)

    out = {
        **{k: v for k, v in source.items() if k not in {"records", "summary"}},
        "run": f"{source['run']}_rejudge",
        "date": date.today().isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": f"rejudged text answers from {args.result.name} with {JUDGE_PROMPT_VERSION}, repeat={args.repeat}",
        "rejudge_of": args.result.name,
        "judge": {**judge_config(), "returned_models": sorted({r["judge_model"] for r in records if r.get("judge_model")})},
        "rejudge": report,
        "git": _git_state(),
        "questions_sha256": _sha256(args.questions),
        "summary": summary,
        "records": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{out['date']}_{out['run']}.json"
    if out_path.exists():
        print(f"{out_path.relative_to(ROOT)} already exists; not overwriting.")
        return 1
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_results_md()
    print(
        f"\nSaved {out_path.relative_to(ROOT)}  "
        f"consistent {report['text_consistent']}/{report['text_judged']}  "
        f"cost ${report['cost_usd']:.5f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
