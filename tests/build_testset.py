"""
Build eval testset from generated FAQ files.

Usage:
    python tests/build_testset.py

Reads all *_faq.txt files from the configured FAQ directory and produces
tests/eval_testset.json with one entry per Q/A pair.

Each entry is marked  "verified": false  — go through the file and flip the
ones that look correct to  "verified": true  before running eval.py.

You can also add your own hard / adversarial cases directly in the JSON file
using the template at the bottom of this file, or by following this format:

{
    "id": "manual_001",
    "question": "...",
    "expected_answer": "...",
    "source_document": "filename-without-extension",
    "source_section": "",          <- leave blank if you don't know exact section
    "category": "limit",           <- limit | entitlement | procedure | exception | definition | negative
    "difficulty": "hard",          <- easy | medium | hard | adversarial
    "verified": true
}

Categories:
    limit        - "What is the maximum X?"
    entitlement  - "Am I eligible for X?"
    procedure    - "How do I apply for X?"
    exception    - "Does policy apply if I am on maternity leave?"
    definition   - "What counts as an eligible dependent?"
    negative     - Answer is genuinely "not covered" or "not mentioned"

Difficulty:
    easy         - exact phrase from the document
    medium       - paraphrased / synonyms used
    hard         - answer requires combining two sections or two documents
    adversarial  - misleading wording, answer is "not covered"
"""

import json
import re
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

FAQ_DIR = Path(r"D:\policy-mgmt\data\vcx_policies\metadata\faq")
OUTPUT_FILE = Path(__file__).parent / "eval_testset.json"

# ── Parser ────────────────────────────────────────────────────────────────────


def parse_faq_file(faq_path: Path) -> list[dict]:
    """Parse Q/A pairs from a *_faq.txt file."""
    text = faq_path.read_text(encoding="utf-8", errors="ignore")

    # Extract source document name from "Source: ..." header
    source_doc = ""
    source_match = re.search(r"^Source:\s*(.+)$", text, re.MULTILINE)
    if source_match:
        raw_source = source_match.group(1).strip()
        source_doc = Path(raw_source).stem  # just the filename stem

    lines = [line.strip() for line in text.splitlines()]
    pairs: list[dict] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("Q:"):
            q = lines[i][2:].strip()
            a = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("A:"):
                a = lines[i + 1][2:].strip()
                # Multi-line answers: keep collecting until next Q: or blank+Q
                j = i + 2
                while j < len(lines) and not lines[j].startswith("Q:"):
                    if lines[j].startswith("A:"):
                        break
                    if lines[j]:
                        a += "\n" + lines[j]
                    j += 1
                i = j
            else:
                i += 1

            if q and a:
                pairs.append({
                    "question": q,
                    "expected_answer": a.strip(),
                    "source_document": source_doc,
                })
        else:
            i += 1

    return pairs


def guess_category(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ("maximum", "minimum", "limit", "cap", "how much", "how many", "what is the")):
        return "limit"
    if any(w in q for w in ("eligible", "entitle", "qualify", "can i", "am i", "do i")):
        return "entitlement"
    if any(w in q for w in ("how do i", "how to", "process", "apply", "steps", "procedure")):
        return "procedure"
    if any(w in q for w in ("exception", "exempt", "unless", "maternity", "leave", "if i am")):
        return "exception"
    if any(w in q for w in ("define", "definition", "what does", "what is meant", "what counts")):
        return "definition"
    if any(w in q for w in ("not covered", "not mentioned", "no mention", "does not")):
        return "negative"
    return "general"


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if not FAQ_DIR.exists():
        print(f"ERROR: FAQ directory not found: {FAQ_DIR}")
        print("Set FAQ_DIR at the top of this script to your faq folder path.")
        sys.exit(1)

    faq_files = sorted(FAQ_DIR.glob("*_faq.txt"))
    if not faq_files:
        print(f"No *_faq.txt files found in {FAQ_DIR}")
        sys.exit(1)

    # Load existing testset so we don't overwrite manually added/verified cases
    existing: dict[str, dict] = {}
    if OUTPUT_FILE.exists():
        for tc in json.loads(OUTPUT_FILE.read_text(encoding="utf-8")):
            existing[tc["id"]] = tc

    test_cases: list[dict] = []
    auto_count = 0

    for faq_path in faq_files:
        pairs = parse_faq_file(faq_path)
        doc_stem = faq_path.stem.replace("_faq", "")

        for idx, pair in enumerate(pairs):
            tc_id = f"faq_{doc_stem}_{idx}"

            if tc_id in existing:
                # Keep the existing entry (may have been verified / edited)
                test_cases.append(existing[tc_id])
            else:
                test_cases.append({
                    "id": tc_id,
                    "question": pair["question"],
                    "expected_answer": pair["expected_answer"],
                    "source_document": pair["source_document"],
                    "source_section": "",
                    "category": guess_category(pair["question"]),
                    "difficulty": "easy",
                    "verified": False,
                })
                auto_count += 1

    # Preserve manually added cases (ids not starting with "faq_")
    for tc_id, tc in existing.items():
        if not tc_id.startswith("faq_"):
            test_cases.append(tc)

    OUTPUT_FILE.write_text(
        json.dumps(test_cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    verified = sum(1 for tc in test_cases if tc.get("verified"))
    print(f"Done.")
    print(f"  Total test cases : {len(test_cases)}")
    print(f"  New (unverified) : {auto_count}")
    print(f"  Verified         : {verified}")
    print(f"  Output           : {OUTPUT_FILE}")
    print()
    print("Next steps:")
    print("  1. Open tests/eval_testset.json")
    print('  2. Set "verified": true for cases that look correct')
    print("  3. Add your own hard/adversarial cases with unique ids like 'manual_001'")
    print("  4. Run:  python tests/eval.py")


if __name__ == "__main__":
    main()
