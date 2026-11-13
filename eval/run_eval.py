"""Run the annual-report QA evaluation.

Examples
--------
    # retrieval metrics only, no API cost
    python eval/run_eval.py --run baseline --retrieval-only

    # full run on the dev split (needs OPENAI_API_KEY; JUDGE_* for text/faithfulness)
    python eval/run_eval.py --run baseline

    # test split, only once at the end of a stage
    python eval/run_eval.py --run stage2_final --split test

Each run writes eval/results/<date>_<run>.json and regenerates eval/RESULTS.md
from every results file. Questions are read, never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # storage paths in .env are relative to the repo root

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from eval import metrics  # noqa: E402
from eval.judge import Judge, judge_config, judge_configured  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_MD = ROOT / "eval" / "RESULTS.md"
MANIFEST_PATH = ROOT / "data" / "manifest.csv"
AUDIT_SHARE = 0.2  # share of judged items listed for human spot-checking


def _price(name: str) -> float:
    try:
        return float(os.getenv(name, "0") or 0)
    except ValueError:
        return 0.0


def _cost(prompt_tokens: int, completion_tokens: int, prefix: str) -> float:
    return (
        prompt_tokens * _price(f"{prefix}_PRICE_INPUT_PER_M")
        + completion_tokens * _price(f"{prefix}_PRICE_OUTPUT_PER_M")
    ) / 1_000_000


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _chunk_record(node) -> dict:
    meta = node.metadata
    return {
        "ticker": meta.get("ticker"),
        "fiscal_year": meta.get("fiscal_year"),
        "page": meta.get("page"),
        "file_name": meta.get("file_name"),
        "score": round(float(node.score or 0.0), 4),
    }


def _in_audit(qid: str) -> bool:
    """Deterministic ~20% sample so the same questions are audited across runs."""
    return int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 100 < AUDIT_SHARE * 100


def evaluate_question(q: dict, pipeline, judge: Judge | None, retrieval_only: bool, rerank_top_n: int) -> dict:
    from rag_pipeline import source_label

    record: dict = {"id": q["id"], "split": q["split"], "answer_type": q["answer_type"], "question": q["question"]}

    start = time.perf_counter()
    ranked = pipeline.retrieve_ranked(q["question"])
    record["retrieval_s"] = round(time.perf_counter() - start, 3)
    chunks = [_chunk_record(n) for n in ranked]
    sent = ranked[:rerank_top_n]
    record["retrieved"] = chunks

    if q["answer_type"] != "not_found":
        record["hit_at_5"] = metrics.hit_at_k(chunks, q, 5)
        record["rr_at_10"] = metrics.reciprocal_rank(chunks, q, 10)
        record["citation_precision"] = metrics.citation_page_precision(chunks[:rerank_top_n], q)

    if retrieval_only:
        record["latency_s"] = record["retrieval_s"]
        return record

    answer = pipeline.answer(q["question"], sent)
    record.update({
        "answer": answer.text,
        "generator_model": answer.model,
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "generation_s": round(answer.latency_s, 3),
        "latency_s": round(record["retrieval_s"] + answer.latency_s, 3),
        "cost_usd": _cost(answer.prompt_tokens, answer.completion_tokens, "LLM"),
    })
    refused = metrics.is_refusal(answer.text)
    record["refused"] = refused

    judge_cost = 0.0
    if q["answer_type"] == "not_found":
        record["correct"] = refused
    elif q["answer_type"] == "number":
        record.update(metrics.score_number(q, answer.text))
    elif judge is not None:  # text
        verdict = judge.grade_text(q["question"], str(q["answer"]), answer.text)
        record["text_label"] = verdict.label
        record["text_reason"] = verdict.reason
        record["judge_model"] = verdict.model
        record["correct"] = verdict.label == "correct"
        judge_cost += _cost(verdict.prompt_tokens, verdict.completion_tokens, "JUDGE")

    if judge is not None and not refused and sent:
        context = "\n\n---\n\n".join(f"[{source_label(n)}]\n{n.get_content()}" for n in sent)
        verdict = judge.grade_faithfulness(context, answer.text)
        record["faithfulness_label"] = verdict.label
        record["faithfulness_reason"] = verdict.reason
        record["unsupported_claims"] = verdict.extra.get("unsupported_claims", [])
        record["judge_model"] = verdict.model
        judge_cost += _cost(verdict.prompt_tokens, verdict.completion_tokens, "JUDGE")

    record["judge_cost_usd"] = judge_cost
    record["audit"] = judge is not None and _in_audit(q["id"])
    return record


def _rate(records: list[dict], key: str, predicate=bool) -> float | None:
    values = [predicate(r[key]) for r in records if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def summarise(records: list[dict]) -> dict:
    answerable = [r for r in records if r["answer_type"] != "not_found"]
    numbers = [r for r in records if r["answer_type"] == "number" and "correct" in r]
    texts = [r for r in records if r["answer_type"] == "text" and "text_label" in r]
    not_found = [r for r in records if r["answer_type"] == "not_found" and "correct" in r]
    answered = [r for r in records if "answer" in r]

    summary = {
        "n": len(records),
        "n_answerable": len(answerable),
        "hit_at_5": metrics.mean(float(r["hit_at_5"]) for r in answerable),
        "mrr_at_10": metrics.mean(r["rr_at_10"] for r in answerable),
        "citation_precision": metrics.mean(r.get("citation_precision") for r in answerable),
        "latency_p50_s": metrics.percentile((r["latency_s"] for r in records), 50),
        "latency_p95_s": metrics.percentile((r["latency_s"] for r in records), 95),
    }
    if answered:
        summary.update({
            "number_accuracy": _rate(numbers, "correct"),
            "n_number": len(numbers),
            "text_correct": _rate(texts, "text_label", lambda v: v == "correct"),
            "text_partial": _rate(texts, "text_label", lambda v: v == "partially_correct"),
            "n_text": len(texts),
            "not_found_accuracy": _rate(not_found, "correct"),
            "n_not_found": len(not_found),
            "false_refusal_rate": _rate([r for r in answered if r["answer_type"] != "not_found"], "refused"),
            "faithfulness_supported": _rate(answered, "faithfulness_label", lambda v: v == "supported"),
            "judge_parse_errors": sum(
                r.get("text_label") == "parse_error" or r.get("faithfulness_label") == "parse_error" for r in answered
            ),
            "avg_prompt_tokens": metrics.mean(r["prompt_tokens"] for r in answered),
            "avg_completion_tokens": metrics.mean(r["completion_tokens"] for r in answered),
            "avg_cost_usd": metrics.mean(r["cost_usd"] for r in answered),
            "judge_cost_usd_total": sum(r.get("judge_cost_usd", 0.0) for r in answered),
            "audit_ids": [r["id"] for r in answered if r.get("audit")],
        })
    return summary


# ── RESULTS.md ────────────────────────────────────────────────────────


def _fmt_pct(v) -> str:
    return "–" if v is None else f"{v * 100:.1f}%"


def _fmt_num(v, digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def write_results_md() -> None:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            runs.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            continue

    lines = [
        "# Evaluation results",
        "",
        "Generated by `eval/run_eval.py` from `eval/results/*.json`. Do not edit by hand.",
        "",
        "Text accuracy shows correct / partially correct (LLM judge). Faithfulness is the share of",
        "non-refused answers whose every claim is supported by the excerpts sent to the model.",
        "Number answers count as correct when any number in the answer matches the gold value",
        "within 0.5% after unit and currency normalisation.",
        "",
        "| Run | Date | Split | N | Hit@5 | MRR@10 | Cite prec. | Number acc. | Text acc. | Not-found acc. | Faithful | p50 / p95 s | $/question | Config |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for path, run in runs:
        cfg = run.get("config", {})
        config = (
            f"embed={cfg.get('embed_model')}; retrieval={cfg.get('retrieval_mode', 'hybrid')}; rerank={cfg.get('rerank_mode')}"
            f"{'(' + cfg['rerank_model'] + ')' if cfg.get('rerank_model') else ''}; "
            f"top_k={cfg.get('top_k')}; top_n={cfg.get('rerank_top_n')}; llm={cfg.get('llm_model')}; "
            f"judge={(run.get('judge') or {}).get('judge_model')}"
        )
        for split, s in run.get("summary", {}).items():
            text = "–" if s.get("text_correct") is None else f"{_fmt_pct(s['text_correct'])} / {_fmt_pct(s.get('text_partial'))}"
            lines.append(
                f"| {run['run']}{' (retrieval only)' if run.get('retrieval_only') else ''} | {run['date']} | {split} | "
                f"{s['n']}{' (limited)' if run.get('limit') else ''} | {_fmt_pct(s['hit_at_5'])} | "
                f"{_fmt_num(s['mrr_at_10'], 3)} | {_fmt_pct(s.get('citation_precision'))} | "
                f"{_fmt_pct(s.get('number_accuracy'))} | {text} | {_fmt_pct(s.get('not_found_accuracy'))} | "
                f"{_fmt_pct(s.get('faithfulness_supported'))} | "
                f"{_fmt_num(s['latency_p50_s'])} / {_fmt_num(s['latency_p95_s'])} | "
                f"{_fmt_num(s.get('avg_cost_usd'), 5)} | {config} |"
            )
    if not runs:
        lines.append("| _no runs yet_ | | | | | | | | | | | | | |")
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="short run name, e.g. baseline or topk20")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--retrieval-only", action="store_true", help="skip generation and judging (no API cost)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test; not for reported numbers)")
    parser.add_argument("--notes", default="", help="what changed in this run")
    args = parser.parse_args()

    out_path = RESULTS_DIR / f"{date.today().isoformat()}_{args.run}.json"
    if out_path.exists():
        print(f"{out_path.relative_to(ROOT)} already exists; pick another --run name so earlier results are kept.")
        return 1

    questions = metrics.load_questions(args.questions)
    if args.split != "all":
        questions = [q for q in questions if q["split"] == args.split]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print(f"No {args.split} questions in {args.questions}.")
        return 1

    import rag_pipeline

    pipeline = rag_pipeline.RAGPipeline()
    if not pipeline.is_ready:
        print("Index not built. Run: DOCS_DIR=data/raw python data_init.py")
        return 1
    if not args.retrieval_only and not pipeline.uses_real_llm:
        print("No OPENAI_API_KEY: mock answers can't be scored. Set a key or pass --retrieval-only.")
        return 1

    judge = None
    if not args.retrieval_only:
        if judge_configured():
            judge = Judge()
        else:
            print("! JUDGE_MODEL not set: text accuracy and faithfulness will be skipped.")

    records = []
    for i, q in enumerate(questions, start=1):
        record = evaluate_question(q, pipeline, judge, args.retrieval_only, rag_pipeline.RERANK_TOP_N)
        records.append(record)
        flag = "" if "correct" not in record else (" ✓" if record["correct"] else " ✗")
        print(f"[{i}/{len(questions)}] {q['id']} hit@5={record.get('hit_at_5', '–')}{flag}")

    splits = sorted({r["split"] for r in records})
    summary = {split: summarise([r for r in records if r["split"] == split]) for split in splits}
    if len(splits) > 1:
        summary["all"] = summarise(records)

    generator_models = sorted({r["generator_model"] for r in records if r.get("generator_model")})
    judge_models = sorted({r["judge_model"] for r in records if r.get("judge_model")})
    result = {
        "run": args.run,
        "date": date.today().isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": args.notes,
        "split": args.split,
        "limit": args.limit,
        "retrieval_only": args.retrieval_only,
        "config": rag_pipeline.effective_config(),
        "judge": {**judge_config(), "returned_models": judge_models} if judge else None,
        "generator_returned_models": generator_models,
        "git": _git_state(),
        "questions_sha256": _sha256(args.questions),
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "summary": summary,
        "records": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_results_md()

    print(f"\nSaved {out_path.relative_to(ROOT)}; updated {RESULTS_MD.relative_to(ROOT)}")
    for split, s in summary.items():
        print(
            f"  {split}: n={s['n']} Hit@5={_fmt_pct(s['hit_at_5'])} MRR@10={_fmt_num(s['mrr_at_10'], 3)}"
            f" number={_fmt_pct(s.get('number_accuracy'))} text={_fmt_pct(s.get('text_correct'))}"
            f" not_found={_fmt_pct(s.get('not_found_accuracy'))} p95={_fmt_num(s['latency_p95_s'])}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
