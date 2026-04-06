import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from policygpt.services.redaction import Redactor


def write_llm_debug_log_pair(
    *,
    log_root: Path | None,
    redactor: Redactor,
    provider: str,
    model_name: str,
    purpose: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    response_text: str,
    error_text: str = "",
) -> None:
    if log_root is None:
        return

    llm_dir = log_root / "llm"
    request_dir = llm_dir / "requests"
    response_dir = llm_dir / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    call_id = str(uuid.uuid4())
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    timestamp = _safe_log_name(timestamp_iso)
    purpose_name = _safe_log_name(purpose)
    file_stem = f"{timestamp}_{purpose_name}_{call_id}"

    request_path = request_dir / f"{file_stem}.txt"
    response_path = response_dir / f"{file_stem}.txt"

    header_lines = [
        f"Call ID: {call_id}",
        f"Timestamp (UTC): {timestamp_iso}",
        f"Purpose: {purpose}",
        f"Model provider: {provider}",
        f"Model name: {model_name}",
        f"Max output tokens: {max_output_tokens}",
    ]

    request_lines = [
        *header_lines,
        f"Matching response file: {response_path.relative_to(llm_dir)}",
        "",
        "=== System Prompt ===",
        redactor.unmask_text(system_prompt).strip() or "(empty)",
        "",
        "=== User Prompt ===",
        redactor.unmask_text(user_prompt).strip() or "(empty)",
    ]

    response_lines = [
        *header_lines,
        f"Matching request file: {request_path.relative_to(llm_dir)}",
        f"Status: {'error' if error_text else 'success'}",
        "",
        "=== Response ===",
        redactor.unmask_text(response_text).strip() or "(empty)",
    ]
    if error_text:
        response_lines.extend(
            [
                "",
                "=== Error ===",
                error_text,
            ]
        )

    request_path.write_text("\n".join(request_lines).strip() + "\n", encoding="utf-8")
    response_path.write_text("\n".join(response_lines).strip() + "\n", encoding="utf-8")


def _safe_log_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    return cleaned.strip("._") or "policygpt_log"
