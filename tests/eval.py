"""
RAG Retrieval Eval Harness for PolicyGPT.

Two modes:

  API mode  (recommended — uses the already-running server, no re-indexing)
  ─────────────────────────────────────────────────────────────────────────
    Start the server first:   python app.py   (or run.bat / run.sh)
    Then run:                 python tests/eval.py --url http://localhost:8000

  Local mode  (no server needed, but re-ingests all documents every run)
  ────────────────────────────────────────────────────────────────────────
    python tests/eval.py --local

Usage:
    python tests/eval.py --url http://localhost:8000
    python tests/eval.py --local
    python tests/eval.py --url http://localhost:8000 --no-answer
    python tests/eval.py --url http://localhost:8000 --category entitlement
    python tests/eval.py --compare tests/results/baseline.json tests/results/after_change.json

Options:
    --url URL           API base URL of running server  (e.g. http://localhost:8000)
    --local             Run in-process without a server (re-ingests on every run)
    --testset PATH      Path to testset JSON  (default: tests/eval_testset.json)
    --output  PATH      Where to write results (default: tests/results/TIMESTAMP.json)
    --compare A B       Compare two result JSON files and exit
    --no-answer         Skip answer generation, retrieval metrics only (API mode only)
    --all               Include unverified cases (default: verified only)
    --category CAT      Run only cases of this category
    --difficulty DIFF   Run only cases of this difficulty
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Add project root to path so imports work when running from anywhere ───────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Default paths ─────────────────────────────────────────────────────────────

DEFAULT_TESTSET = Path(__file__).parent / "eval_testset.json"
RESULTS_DIR = Path(__file__).parent / "results"

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    tc_id: str
    question: str
    expected_answer: str
    source_document: str
    source_section: str
    category: str
    difficulty: str

    # Retrieval
    retrieved_docs: list[str] = field(default_factory=list)
    retrieved_sections: list[str] = field(default_factory=list)
    hit_at_1: bool = False
    hit_at_3: bool = False
    reciprocal_rank: float = 0.0

    # Answer
    generated_answer: str = ""
    via_faq_fastpath: bool = False

    # Timing
    latency_ms: float = 0.0
    error: Optional[str] = None


# ── Matching helpers ──────────────────────────────────────────────────────────


def _section_hit(tc: dict, doc_names: list[str], section_titles: list[str]) -> bool:
    """Return True if expected doc+section appears anywhere in the paired lists.

    source_document in the testset is the HTML filename stem
    (e.g. "anti-trust-and-anti-competitive-business-practices-policy-").
    The retrieved doc name is the human title from the bot
    (e.g. "Anti-Trust and Anti-Competitive Business Practices Policy").

    We match if enough tokens from the expected stem appear in the retrieved title.
    """
    expected_doc = (tc.get("source_document") or "").lower().strip()
    expected_sec = (tc.get("source_section") or "").lower().strip()

    if not expected_doc:
        return True  # no doc constraint — any retrieval counts

    expected_tokens = {t for t in re.split(r"[\s\-_]+", expected_doc) if len(t) >= 4}

    for doc, sec in zip(doc_names, section_titles):
        doc_lower = doc.lower()
        token_overlap = sum(1 for t in expected_tokens if t in doc_lower)
        doc_match = (
            expected_doc in doc_lower
            or (expected_tokens and token_overlap >= max(1, len(expected_tokens) // 2))
        )
        if not doc_match:
            continue
        if not expected_sec or expected_sec in sec.lower():
            return True
    return False


def _reciprocal_rank(tc: dict, doc_names: list[str], section_titles: list[str]) -> float:
    for rank, (doc, sec) in enumerate(zip(doc_names, section_titles), start=1):
        if _section_hit(tc, [doc], [sec]):
            return 1.0 / rank
    return 0.0


# ── API mode ──────────────────────────────────────────────────────────────────


def _api_ask(base_url: str, question: str, session: "requests.Session") -> tuple[str, list[str], list[str]]:
    """POST to /api/chat and return (answer, retrieved_docs, retrieved_sections)."""
    import requests  # only needed in API mode

    # Create a fresh thread
    thread_resp = session.post(f"{base_url}/api/threads", timeout=10)
    thread_resp.raise_for_status()
    thread_id = thread_resp.json()["thread_id"]

    chat_resp = session.post(
        f"{base_url}/api/chat",
        json={"message": question, "thread_id": thread_id},
        timeout=120,
    )
    chat_resp.raise_for_status()
    data = chat_resp.json()

    answer = data.get("answer", "")
    sources = data.get("thread", {}).get("sources", [])
    retrieved_docs = [s.get("document_title", "") for s in sources]
    retrieved_sections = [s.get("section_title", "") for s in sources]
    return answer, retrieved_docs, retrieved_sections


def run_eval_api(
    base_url: str,
    testset_path: Path,
    output_path: Path,
    only_verified: bool = True,
    generate_answers: bool = True,
    filter_category: str = "",
    filter_difficulty: str = "",
    user_id: str = "",
) -> list[EvalResult]:
    import requests

    test_cases = _load_test_cases(testset_path, only_verified, filter_category, filter_difficulty)

    # Wait for server to be ready
    print(f"Checking server at {base_url} ...")
    _wait_for_server(base_url)
    print(f"Server ready. Running {len(test_cases)} test cases...\n")

    session = requests.Session()
    # Set user_id cookie so permission filtering works the same as in production
    if user_id:
        session.cookies.set("user_id", user_id)
    results: list[EvalResult] = []

    for i, tc in enumerate(test_cases, start=1):
        result = _make_result(tc)
        print(f"[{i:>3}/{len(test_cases)}] {tc['id']}")
        print(f"         Q: {tc['question'][:90]}")

        try:
            t0 = time.perf_counter()

            if generate_answers:
                answer, retrieved_docs, retrieved_sections = _api_ask(base_url, tc["question"], session)
                result.generated_answer = answer
                result.retrieved_docs = retrieved_docs
                result.retrieved_sections = retrieved_sections
            else:
                # Retrieval-only: still call the API but discard the answer
                _, retrieved_docs, retrieved_sections = _api_ask(base_url, tc["question"], session)
                result.retrieved_docs = retrieved_docs
                result.retrieved_sections = retrieved_sections

            result.latency_ms = (time.perf_counter() - t0) * 1000
            _score(result, tc)
            _print_result_line(result)

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            print(f"         ERROR: {result.error}")

        print()
        results.append(result)

    _save_results(results, output_path)
    _print_summary(results)
    return results


def _wait_for_server(base_url: str, retries: int = 5, delay: float = 2.0) -> None:
    import requests
    for attempt in range(retries):
        try:
            r = requests.get(f"{base_url}/api/health", timeout=5)
            data = r.json()
            status = data.get("status", "")
            if status == "ready":
                return
            print(f"  Server status: {status} — waiting...")
        except Exception:
            print(f"  Server not reachable (attempt {attempt + 1}/{retries}) — waiting...")
        time.sleep(delay)
    print("WARNING: Server may not be fully ready, proceeding anyway.")


# ── Local mode ────────────────────────────────────────────────────────────────


def run_eval_local(
    testset_path: Path,
    output_path: Path,
    only_verified: bool = True,
    generate_answers: bool = True,
    filter_category: str = "",
    filter_difficulty: str = "",
) -> list[EvalResult]:
    from policygpt.factory import create_ready_bot

    test_cases = _load_test_cases(testset_path, only_verified, filter_category, filter_difficulty)

    print("Loading PolicyGPT bot (ingesting documents — this may take a while)...")
    bot = create_ready_bot()
    print(f"Bot ready. Running {len(test_cases)} test cases...\n")

    results: list[EvalResult] = []

    for i, tc in enumerate(test_cases, start=1):
        result = _make_result(tc)
        print(f"[{i:>3}/{len(test_cases)}] {tc['id']}")
        print(f"         Q: {tc['question'][:90]}")

        try:
            t0 = time.perf_counter()
            thread_id = bot.new_thread()

            if generate_answers:
                chat_result = bot.chat(thread_id, tc["question"])
                result.generated_answer = chat_result.answer
                result.via_faq_fastpath = getattr(chat_result, "via_faq_fastpath", False)
                for src in chat_result.sources:
                    result.retrieved_docs.append(src.document_title)
                    result.retrieved_sections.append(src.section_title)
            else:
                # Retrieval only — skip LLM generation
                from policygpt.core.retrieval.query_analyzer import QueryAnalyzer  # noqa
                query_vec = bot.corpus.embed_text(tc["question"])
                qa = bot.query_analyzer.analyze(
                    user_question=tc["question"],
                    active_document_titles=[],
                    candidate_documents=list(bot.corpus.documents.values()),
                    entity_lookup=bot.corpus.entity_lookup,
                )
                top_docs = bot.corpus.retrieve_top_docs(query_vec, qa)
                top_sections = bot.corpus.retrieve_top_sections(query_vec, qa, top_docs)
                for sec, _score_val in top_sections[:3]:
                    doc = bot.corpus.documents.get(sec.doc_id)
                    result.retrieved_docs.append(doc.title if doc else sec.doc_id)
                    result.retrieved_sections.append(sec.title)

            result.latency_ms = (time.perf_counter() - t0) * 1000
            _score(result, tc)
            _print_result_line(result)

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            print(f"         ERROR: {result.error}")

        print()
        results.append(result)

    _save_results(results, output_path)
    _print_summary(results)
    return results


# ── Shared helpers ────────────────────────────────────────────────────────────


def _load_test_cases(
    testset_path: Path,
    only_verified: bool,
    filter_category: str,
    filter_difficulty: str,
) -> list[dict]:
    test_cases: list[dict] = json.loads(testset_path.read_text(encoding="utf-8"))
    if only_verified:
        test_cases = [tc for tc in test_cases if tc.get("verified")]
    if filter_category:
        test_cases = [tc for tc in test_cases if tc.get("category") == filter_category]
    if filter_difficulty:
        test_cases = [tc for tc in test_cases if tc.get("difficulty") == filter_difficulty]
    if not test_cases:
        print("No test cases match the filters (did you set verified=true in the testset?).")
        sys.exit(1)
    return test_cases


def _make_result(tc: dict) -> EvalResult:
    return EvalResult(
        tc_id=tc["id"],
        question=tc["question"],
        expected_answer=tc.get("expected_answer", ""),
        source_document=tc.get("source_document", ""),
        source_section=tc.get("source_section", ""),
        category=tc.get("category", ""),
        difficulty=tc.get("difficulty", ""),
    )


def _score(result: EvalResult, tc: dict) -> None:
    result.hit_at_1 = _section_hit(tc, result.retrieved_docs[:1], result.retrieved_sections[:1])
    result.hit_at_3 = _section_hit(tc, result.retrieved_docs[:3], result.retrieved_sections[:3])
    result.reciprocal_rank = _reciprocal_rank(tc, result.retrieved_docs[:3], result.retrieved_sections[:3])


def _print_result_line(result: EvalResult) -> None:
    status = "HIT " if result.hit_at_1 else "MISS"
    print(f"         [{status}]  latency={result.latency_ms:.0f}ms  rr={result.reciprocal_rank:.2f}")
    if result.retrieved_docs:
        print(f"         top-1: {result.retrieved_docs[0]} / {result.retrieved_sections[0]}")


# ── Summary & reporting ───────────────────────────────────────────────────────


def _print_summary(results: list[EvalResult]) -> None:
    valid = [r for r in results if not r.error]
    n = len(valid)
    errors = len(results) - n
    if not n:
        print("No valid results to summarise.")
        return

    hit1 = sum(r.hit_at_1 for r in valid) / n
    hit3 = sum(r.hit_at_3 for r in valid) / n
    mrr = sum(r.reciprocal_rank for r in valid) / n
    latencies = sorted(r.latency_ms for r in valid)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    faq_hits = sum(r.via_faq_fastpath for r in valid)

    print("=" * 52)
    print(f"  Test cases : {n}  (errors: {errors})")
    print(f"  Hit@1      : {hit1:.1%}")
    print(f"  Hit@3      : {hit3:.1%}")
    print(f"  MRR        : {mrr:.3f}")
    print(f"  FAQ hits   : {faq_hits} ({faq_hits/n:.1%})")
    print(f"  Latency    : p50={p50:.0f}ms  p95={p95:.0f}ms")
    print("=" * 52)

    # Per-category breakdown
    categories = sorted({r.category for r in valid if r.category})
    if len(categories) > 1:
        print("\nBreakdown by category:")
        for cat in categories:
            cat_r = [r for r in valid if r.category == cat]
            print(f"  {cat:<14}  n={len(cat_r):<4}  Hit@1={sum(r.hit_at_1 for r in cat_r)/len(cat_r):.1%}")

    # Per-difficulty breakdown
    difficulties = sorted({r.difficulty for r in valid if r.difficulty})
    if len(difficulties) > 1:
        print("\nBreakdown by difficulty:")
        for diff in difficulties:
            diff_r = [r for r in valid if r.difficulty == diff]
            print(f"  {diff:<14}  n={len(diff_r):<4}  Hit@1={sum(r.hit_at_1 for r in diff_r)/len(diff_r):.1%}")

    # Misses
    misses = [r for r in valid if not r.hit_at_3]
    if misses:
        print(f"\nMisses (not in top 3) — {len(misses)} cases:")
        for r in misses[:10]:
            print(f"  [{r.tc_id}]  {r.question[:70]}")
            print(f"    expected : {r.source_document}")
            print(f"    got      : {r.retrieved_docs[:1]}")
        if len(misses) > 10:
            print(f"  ... and {len(misses) - 10} more (see results JSON)")

    print()


def _save_results(results: list[EvalResult], output_path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = [vars(r) for r in results]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved → {output_path}\n")


# ── Compare two runs ──────────────────────────────────────────────────────────


def compare_runs(path_a: Path, path_b: Path) -> None:
    def load(p: Path) -> dict[str, dict]:
        return {r["tc_id"]: r for r in json.loads(p.read_text(encoding="utf-8"))}

    a, b = load(path_a), load(path_b)
    common = set(a) & set(b)
    if not common:
        print("No common test case IDs between the two result files.")
        return

    regressions = [tc_id for tc_id in common if a[tc_id]["hit_at_1"] and not b[tc_id]["hit_at_1"]]
    improvements = [tc_id for tc_id in common if not a[tc_id]["hit_at_1"] and b[tc_id]["hit_at_1"]]

    def hit1(results: dict) -> float:
        valid = [r for r in results.values() if not r.get("error")]
        return sum(r["hit_at_1"] for r in valid) / len(valid) if valid else 0.0

    def mrr_score(results: dict) -> float:
        valid = [r for r in results.values() if not r.get("error")]
        return sum(r["reciprocal_rank"] for r in valid) / len(valid) if valid else 0.0

    a_hit1, b_hit1 = hit1(a), hit1(b)
    a_mrr, b_mrr = mrr_score(a), mrr_score(b)

    print(f"\nComparing: {path_a.name}  vs  {path_b.name}")
    print("=" * 52)
    print(f"  Hit@1 : {a_hit1:.1%}  →  {b_hit1:.1%}  (Δ {b_hit1 - a_hit1:+.1%})")
    print(f"  MRR   : {a_mrr:.3f}  →  {b_mrr:.3f}  (Δ {b_mrr - a_mrr:+.3f})")
    print("=" * 52)

    if regressions:
        print(f"\nRegressions ({len(regressions)}) — Hit@1 in A, MISS in B:")
        for tc_id in regressions:
            print(f"  {tc_id}: {a[tc_id]['question'][:70]}")

    if improvements:
        print(f"\nImprovements ({len(improvements)}) — MISS in A, Hit@1 in B:")
        for tc_id in improvements:
            print(f"  {tc_id}: {b[tc_id]['question'][:70]}")

    if not regressions and not improvements:
        print("\nNo changes in Hit@1 for common cases.")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="PolicyGPT RAG eval harness")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--url", metavar="URL", help="API base URL of running server (e.g. http://localhost:8000)")
    mode.add_argument("--local", action="store_true", help="Run in-process (re-ingests documents)")
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    parser.add_argument("--no-answer", action="store_true", help="Retrieval metrics only")
    parser.add_argument("--all", dest="all_cases", action="store_true", help="Include unverified cases")
    parser.add_argument("--category", default="", help="Filter by category")
    parser.add_argument("--difficulty", default="", help="Filter by difficulty")
    parser.add_argument("--user-id", default="", help="user_id to send as cookie (required when hybrid search is enabled)")
    args = parser.parse_args()

    if args.compare:
        compare_runs(Path(args.compare[0]), Path(args.compare[1]))
        return

    if not args.testset.exists():
        print(f"Testset not found: {args.testset}")
        print("Run:  python tests/build_testset.py  to generate it first.")
        sys.exit(1)

    if not args.url and not args.local:
        parser.error("Specify --url http://localhost:8000  or  --local")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (RESULTS_DIR / f"eval_{timestamp}.json")
    kwargs = dict(
        testset_path=args.testset,
        output_path=output_path,
        only_verified=not args.all_cases,
        generate_answers=not args.no_answer,
        filter_category=args.category,
        filter_difficulty=args.difficulty,
    )

    if args.url:
        run_eval_api(base_url=args.url.rstrip("/"), user_id=args.user_id, **kwargs)
    else:
        run_eval_local(**kwargs)


if __name__ == "__main__":
    main()
