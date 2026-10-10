#!/usr/bin/env python3
"""Score the pipeline on tests/fixtures/eval_cases.json — retrieval and (optionally) the LLM.

Unlike the pytest eval (which stops at the first miss), this scores every case and
prints per-question latency, so profiles can be compared:

  python scripts/eval_llm.py                                # default profile, retrieval + LLM
  python scripts/eval_llm.py --retrieval-only
  python scripts/eval_llm.py --cases tests/fixtures/hard_cases.json   # tricky interactions
  MTG_JUDGE_CONFIG=config/pi.yaml python scripts/eval_llm.py   # Pi model + budgets
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from src.app import build_pipeline
from src.config import load_config


def _parent(section: str) -> str:
    m = re.match(r"(\d+\.\d+)", section or "")
    return m.group(1) if m else section or ""


def section_matches(expected: str, sections: set[str]) -> bool:
    """'405' matches 405.2; '702.16' matches 702.16 (not 702.160)."""
    return any(s == expected or s.startswith(expected + ".") for s in sections)


def score_retrieval(case: dict, route: dict, chunks: list[dict]) -> dict[str, bool]:
    text = " ".join(c.get("text", "") + " " + c.get("name", "") for c in chunks).lower()
    sources = {c.get("source") for c in chunks}
    checks = {
        "sources": all(s in sources for s in case["expected_sources"]),
        "keywords": all(k.lower() in text for k in case.get("expected_keywords", [])),
    }
    if "expected_sections_any" in case:
        sections = {_parent(c.get("parent_section") or c.get("section", "")) for c in chunks}
        checks["sections"] = any(section_matches(s, sections) for s in case["expected_sections_any"])
    if "expected_cards" in case:
        checks["cards"] = all(card.lower() in text for card in case["expected_cards"])
    return checks


def grade_answer(case: dict, answer: str) -> bool | None:
    """`verdict` ("yes"/"no") must lead the answer, every `answer_any` group must have at
    least one phrase in it, and no `answer_none` phrase may appear. None = ungraded."""
    groups, verdict, forbidden = case.get("answer_any", []), case.get("verdict"), case.get("answer_none", [])
    if not groups and not verdict:
        return None
    text = answer.lower().lstrip(" \"'")
    if verdict and not re.match(rf"{verdict}\b", text):
        return False
    if any(phrase.lower() in text for phrase in forbidden):
        return False
    return all(any(phrase.lower() in text for phrase in group) for group in groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--cases", default="tests/fixtures/eval_cases.json")
    parser.add_argument("--show", action="store_true", help="print every answer (misses are always printed)")
    parser.add_argument("--repeat", type=int, default=1, help="run the cases N times (LLM sampling is noisy)")
    args = parser.parse_args()

    config = load_config(args.config)
    retriever, router, llm = build_pipeline(config)
    cases = json.loads(Path(args.cases).read_text()) * args.repeat
    think = "" if config.get("think") is None else f", think={config['think']}"
    print(f"Profile: LLM {config['llm_model']}{think}, k={config['k']}, context ≤{config['max_context_chars']} chars\n")

    passed, cited, t_retrieve, t_llm = 0, 0, [], []
    graded, correct = 0, 0
    by_topic: dict[str, list[int]] = {}  # topic → [correct, graded]
    prompt_tokens, output_tokens = [], []
    for case in cases:
        q = case["question"]
        t0 = time.perf_counter()
        route = router.route(q)
        chunks = retriever.retrieve(q, route=route)
        t1 = time.perf_counter()
        checks = score_retrieval(case, route, chunks)
        ok = all(checks.values())
        passed += ok
        t_retrieve.append(t1 - t0)

        line = f"{'✓' if ok else '✗'} {q[:58]:58} retr {t1 - t0:5.2f}s"
        if not ok:
            line += f"  missed: {[k for k, v in checks.items() if not v]}"

        if not args.retrieval_only:
            ruling = llm.answer(q, chunks)
            t2 = time.perf_counter()
            t_llm.append(t2 - t1)
            prompt_tokens.append(llm.last_stats["prompt_tokens"])
            output_tokens.append(llm.last_stats["output_tokens"])
            expected = case.get("expected_sections_any", [])
            cites = {_parent(m) for m in re.findall(r"\d{3}\.\d+[a-z]{0,2}", " ".join(ruling.citations))}
            has_cite = not expected or any(section_matches(s, cites) for s in expected)
            cited += has_cite
            line += f"  llm {t2 - t1:5.1f}s [{ruling.confidence}]{'' if has_cite else ' (expected cite missing)'}"
            verdict = grade_answer(case, ruling.answer)
            if verdict is not None:
                graded += 1
                correct += verdict
                tally = by_topic.setdefault(case.get("topic", "-"), [0, 0])
                tally[0] += verdict
                tally[1] += 1
                line += "  answer ✓" if verdict else "  answer ✗"
            if args.show or verdict is False:
                line += f"\n    → {ruling.answer}  {ruling.citations}"
                if verdict is False and case.get("truth"):
                    line += f"\n    expected: {case['truth']}"
        print(line)

    n = len(cases)
    print(f"\nRetrieval: {passed}/{n} cases fully passed · median {statistics.median(t_retrieve):.2f}s")
    if t_llm:
        print(f"LLM: {cited}/{n} cite an expected section · median {statistics.median(t_llm):.1f}s · max {max(t_llm):.1f}s")
        print(f"Tokens: median prompt {statistics.median(prompt_tokens):.0f} · median output {statistics.median(output_tokens):.0f}"
              f" · max output {max(output_tokens)}")
    if graded:
        print(f"Answers: {correct}/{graded} graded answers correct")
        if len(by_topic) > 1:
            print("By topic: " + " · ".join(f"{t} {c}/{n}" for t, (c, n) in sorted(by_topic.items())))


if __name__ == "__main__":
    main()
