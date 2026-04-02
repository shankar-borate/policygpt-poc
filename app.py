from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock, Thread
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from policy_gpt_poc import Config, PolicyGPTBot, create_ready_bot


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
INDEX_HEAD_INJECTION = """
<style>
    .reference-panel {
        display: none !important;
    }

    .message-reference-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }

    .message-reference-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border: 1px solid #e5e7eb;
        border-radius: 999px;
        color: #6b7280;
        background: #f3f4f6;
        font-size: 0.74rem;
        font-style: italic;
        line-height: 1.2;
        white-space: nowrap;
    }
</style>
"""
INDEX_BODY_INJECTION = """
<script>
(() => {
    function buildReferenceRow(rawText) {
        const text = String(rawText || "").trim();
        if (!text.toLowerCase().startsWith("reference:")) {
            return null;
        }

        const values = text
            .slice("Reference:".length)
            .split(",")
            .map((value) => value.trim())
            .filter(Boolean);

        if (!values.length) {
            return null;
        }

        const row = document.createElement("div");
        row.className = "message-reference-row";

        values.forEach((value) => {
            const pill = document.createElement("span");
            pill.className = "message-reference-pill";
            pill.textContent = value;
            row.appendChild(pill);
        });

        return row;
    }

    function enhanceMessageReferences() {
        document.querySelectorAll(".message.assistant .message-body").forEach((body) => {
            if (body.dataset.referenceEnhanced === "true") {
                return;
            }

            const paragraphs = body.querySelectorAll(":scope > p");
            const lastParagraph = paragraphs[paragraphs.length - 1];
            if (!lastParagraph) {
                body.dataset.referenceEnhanced = "true";
                return;
            }

            const referenceRow = buildReferenceRow(lastParagraph.textContent);
            if (referenceRow) {
                lastParagraph.replaceWith(referenceRow);
            }

            body.dataset.referenceEnhanced = "true";
        });
    }

    function startReferenceEnhancer() {
        const messages = document.getElementById("messages");
        if (!messages) {
            return;
        }

        const observer = new MutationObserver(() => {
            enhanceMessageReferences();
        });

        observer.observe(messages, { childList: true, subtree: true });
        enhanceMessageReferences();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", startReferenceEnhancer, { once: true });
    } else {
        startReferenceEnhancer();
    }
})();
</script>
"""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: Optional[str] = None


class RuntimeState:
    def __init__(self) -> None:
        self.lock = RLock()
        self.bot: Optional[PolicyGPTBot] = None
        self.status = "starting"
        self.error: Optional[str] = None
        self.document_folder = Config.DOCUMENT_FOLDER
        self.worker: Optional[Thread] = None
        self.indexing_processed_files = 0
        self.indexing_total_files = 0
        self.indexing_current_file: Optional[str] = None
        self.document_count = 0
        self.section_count = 0

    def _update_progress(
        self,
        processed_files: int,
        total_files: int,
        current_file: Optional[str],
        document_count: int,
        section_count: int,
    ) -> None:
        with self.lock:
            self.status = "in_progress"
            self.error = None
            self.indexing_processed_files = processed_files
            self.indexing_total_files = total_files
            self.indexing_current_file = current_file
            self.document_count = document_count
            self.section_count = section_count

    def _initialize_worker(self) -> None:
        try:
            print(
                f"[Policy GPT] Indexing started for {self.document_folder}",
                flush=True,
            )
            bot = create_ready_bot(
                self.document_folder,
                progress_callback=self._update_progress,
            )
            with self.lock:
                self.bot = bot
                self.status = "ready"
                self.error = None
                self.indexing_processed_files = self.indexing_total_files
                self.indexing_current_file = None
                self.document_count = len(bot.documents)
                self.section_count = len(bot.sections)
            print(
                f"[Policy GPT] Indexing complete: {len(bot.documents)} documents, {len(bot.sections)} sections.",
                flush=True,
            )
        except Exception as exc:
            with self.lock:
                self.bot = None
                self.status = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                self.indexing_current_file = None
            print(
                f"[Policy GPT] Indexing failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            with self.lock:
                self.worker = None

    def start_indexing(self) -> None:
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                return

            self.bot = None
            self.status = "in_progress"
            self.error = None
            self.indexing_processed_files = 0
            self.indexing_total_files = 0
            self.indexing_current_file = None
            self.document_count = 0
            self.section_count = 0
            self.worker = Thread(target=self._initialize_worker, daemon=True)
            self.worker.start()

    def progress_detail(self) -> str:
        total_files = self.indexing_total_files
        if total_files <= 0:
            return "Indexing is in progress."

        processed_files = min(self.indexing_processed_files, total_files)
        current_step = self.indexing_current_file or "the next indexing step"
        return f"Indexing is in progress ({processed_files}/{total_files}). Current step: {current_step}"

    def progress_payload(self) -> dict:
        total_files = self.indexing_total_files
        processed_files = min(self.indexing_processed_files, total_files) if total_files else 0
        percent = round((processed_files / total_files) * 100, 1) if total_files else 0.0
        return {
            "processed_files": processed_files,
            "total_files": total_files,
            "current_file": self.indexing_current_file,
            "percent": percent,
        }


runtime = RuntimeState()


def require_bot() -> PolicyGPTBot:
    if runtime.bot is None or runtime.status != "ready":
        if runtime.status == "in_progress":
            detail = runtime.progress_detail()
        else:
            detail = runtime.error or "Policy GPT is still initializing."
        raise HTTPException(status_code=503, detail=detail)
    return runtime.bot


def build_preview(text: str, limit: int = 96) -> str:
    compact = " ".join(text.split())
    if not compact:
        return ""
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def serialize_message(message) -> dict:
    return {
        "role": message.role,
        "content": message.content,
    }


def serialize_source(source) -> dict:
    return {
        "document_title": source.document_title,
        "section_title": source.section_title,
        "source_path": source.source_path,
        "file_name": Path(source.source_path).name,
        "score": round(source.score, 4),
    }


def serialize_thread_summary(thread) -> dict:
    preview_source = thread.display_messages[-1].content if thread.display_messages else ""
    return {
        "thread_id": thread.thread_id,
        "title": thread.title,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
        "message_count": len(thread.display_messages),
        "preview": build_preview(preview_source),
    }


def serialize_thread_detail(thread) -> dict:
    return {
        **serialize_thread_summary(thread),
        "messages": [serialize_message(message) for message in thread.display_messages],
        "sources": [serialize_source(source) for source in thread.last_answer_sources],
        "conversation_summary": thread.conversation_summary,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start_indexing()
    yield


app = FastAPI(title="Policy GPT", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def inject_index_overrides(html: str) -> str:
    with_head = html.replace("</head>", f"{INDEX_HEAD_INJECTION}</head>")
    return with_head.replace("</body>", f"{INDEX_BODY_INJECTION}</body>")


@app.get("/")
def index() -> HTMLResponse:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(inject_index_overrides(html))


@app.get("/api/health")
def health() -> dict:
    with runtime.lock:
        bot = runtime.bot
        return {
            "status": runtime.status,
            "error": runtime.error,
            "document_folder": runtime.document_folder,
            "document_count": runtime.document_count,
            "section_count": runtime.section_count,
            "thread_count": len(bot.threads) if bot else 0,
            "progress": runtime.progress_payload(),
        }


@app.get("/api/threads")
def list_threads() -> dict:
    with runtime.lock:
        bot = require_bot()
        return {
            "items": [serialize_thread_summary(thread) for thread in bot.list_threads()],
        }


@app.post("/api/threads")
def create_thread() -> dict:
    with runtime.lock:
        bot = require_bot()
        thread_id = bot.new_thread()
        return serialize_thread_detail(bot.get_thread(thread_id))


@app.get("/api/threads/{thread_id}")
def get_thread(thread_id: str) -> dict:
    with runtime.lock:
        bot = require_bot()
        thread = bot.threads.get(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        return serialize_thread_detail(thread)


@app.post("/api/threads/{thread_id}/reset")
def reset_thread(thread_id: str) -> dict:
    with runtime.lock:
        bot = require_bot()
        if thread_id not in bot.threads:
            raise HTTPException(status_code=404, detail="Thread not found.")
        bot.reset_thread(thread_id)
        return serialize_thread_detail(bot.get_thread(thread_id))


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    with runtime.lock:
        bot = require_bot()
        thread_id = request.thread_id or bot.new_thread()
        result = bot.chat(thread_id=thread_id, user_question=message)
        thread = bot.get_thread(result.thread_id)
        return {
            "thread": serialize_thread_detail(thread),
            "answer": result.answer,
        }


if __name__ == "__main__":
    import uvicorn

    print(
        "[Policy GPT] Starting server on http://127.0.0.1:8010 (plain HTTP, not HTTPS).",
        flush=True,
    )
    uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=False)
