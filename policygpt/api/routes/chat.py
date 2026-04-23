from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from policygpt.config import Config
from policygpt.config.user_profiles import resolve_user_profile
from policygpt.constants import FileExtension
from policygpt.core.document_links import build_document_open_url
from policygpt.api.renderers.document_viewer import DocumentViewerRenderer
from policygpt.api.runtime import ServerRuntime
from policygpt.api.renderers.ui import WebUIRenderer
from policygpt.models import ThreadState


def _resolve_user_id(http_request: Request, query_user_id: str | None = None) -> str:
    """Return user_id from query param first, then cookie, then empty string."""
    if query_user_id:
        return query_user_id
    return http_request.cookies.get("user_id", "")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None


class PolicyApiServer:
    def __init__(self, config: Config, web_dir: Path) -> None:
        self.config = config
        self.web_dir = web_dir
        self.runtime = ServerRuntime(config)
        self.ui_renderer = WebUIRenderer(web_dir)
        self.document_viewer = DocumentViewerRenderer(web_dir)

    def build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_: FastAPI):
            self.runtime.start_indexing()
            yield

        app = FastAPI(title="Policy GPT", lifespan=lifespan)
        app.mount("/static", StaticFiles(directory=str(self.web_dir)), name="static")

        app.add_api_route("/", self.index, methods=["GET"], response_class=HTMLResponse)
        app.add_api_route("/api/domain", self.domain_ui, methods=["GET"])
        app.add_api_route("/api/health", self.health, methods=["GET"])
        app.add_api_route("/api/usage", self.usage, methods=["GET"])
        app.add_api_route("/api/threads", self.list_threads, methods=["GET"])
        app.add_api_route("/api/threads", self.create_thread, methods=["POST"])
        app.add_api_route("/api/threads/{thread_id}", self.get_thread, methods=["GET"])
        app.add_api_route("/api/threads/{thread_id}/reset", self.reset_thread, methods=["POST"])
        app.add_api_route("/api/chat", self.chat, methods=["POST"])
        app.add_api_route("/api/search", self.search, methods=["GET"])
        app.add_api_route("/api/documents/open", self.open_document, methods=["GET"], response_class=FileResponse)
        app.add_api_route("/api/documents/view", self.view_document, methods=["GET"], response_class=HTMLResponse)
        return app

    def index(self) -> HTMLResponse:
        return HTMLResponse(
            self.ui_renderer.render_index(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    def domain_ui(self) -> dict:
        profile = self.config.domain_profile
        return {
            "assistant_label": profile.ui_assistant_label,
            "eyebrow": profile.ui_eyebrow,
            "description": profile.ui_description,
            "sidebar_title": profile.ui_sidebar_title,
            "sidebar_subtitle": profile.ui_sidebar_subtitle,
            "search_placeholder": profile.ui_search_placeholder,
            "input_placeholder": profile.ui_input_placeholder,
            "prompt_chips": [
                {"label": label, "prompt": prompt}
                for label, prompt in profile.ui_prompt_chips
            ],
        }

    def health(self) -> dict:
        with self.runtime.lock:
            bot = self.runtime.bot
            status = self.runtime.status
            # "ingesting" is a sub-state of ready — UI treats both as operational
            display_status = "ready" if status == "ingesting" else status
        return {
            "status": display_status,
            "ingesting": status == "ingesting",
            "error": self.runtime.error,
            "document_folder": self.runtime.config.storage.document_folder,
            "document_count": self.runtime.get_document_count(),
            "section_count": self.runtime.get_section_count(),
            "thread_count": len(bot.threads) if bot else 0,
            "progress": self.runtime.progress_payload(),
        }

    def usage(self) -> dict:
        with self.runtime.lock:
            return self.runtime.usage_payload()

    def search(
        self,
        q: str,
        http_request: Request,
        page: int = 1,
        size: int = 10,
        user_id: str | None = None,
    ) -> dict:
        q = q.strip()
        if not q:
            raise HTTPException(status_code=422, detail="Query cannot be empty.")

        user_id = _resolve_user_id(http_request, user_id)
        if not user_id and self.config.search.hybrid_search_enabled:
            raise HTTPException(status_code=401, detail="user_id cookie or query param is required.")

        page = max(1, page)
        size = min(max(1, size), 50)

        with self.runtime.lock:
            bot = self.runtime.require_bot()
            vector_store = bot.corpus._vector_store
            if vector_store is None:
                raise HTTPException(
                    status_code=503,
                    detail="Search requires OpenSearch to be configured (hybrid_search_enabled=True).",
                )
            try:
                raw = vector_store.search_documents(
                    query_text=q,
                    user_id=user_id,
                    page=page,
                    size=size,
                )
            except NotImplementedError:
                raise HTTPException(
                    status_code=503,
                    detail="Search is not supported by the current vector store backend.",
                )

        results = [
            {
                "document_title": r["document_title"],
                "section_title":  r["section_title"],
                "snippet":        r["snippet"],
                "score":          r["score"],
                "document_url": build_document_open_url(
                    self.config.storage.public_base_url,
                    r["source_path"],
                ),
            }
            for r in raw["results"]
        ]
        return {
            "query":   q,
            "total":   raw["total"],
            "page":    raw["page"],
            "size":    raw["size"],
            "results": results,
        }

    def list_threads(self, http_request: Request, user_id: str | None = None) -> dict:
        user_id = _resolve_user_id(http_request, user_id)
        with self.runtime.lock:
            bot = self.runtime.require_bot()
            return {
                "items": [
                    self.serialize_thread_summary(t)
                    for t in bot.list_threads(user_id=user_id)
                ],
            }

    def create_thread(self, http_request: Request, user_id: str | None = None) -> dict:
        user_id = _resolve_user_id(http_request, user_id)
        with self.runtime.lock:
            bot = self.runtime.require_bot()
            thread_id = bot.new_thread(user_id=user_id)
            thread = bot.conversations.get_thread_for_display(thread_id)
            return self.serialize_thread_detail(thread or ThreadState(thread_id=thread_id))

    def get_thread(self, thread_id: str, http_request: Request) -> dict:
        with self.runtime.lock:
            bot = self.runtime.require_bot()
            thread = bot.conversations.get_thread_for_display(thread_id)
            if thread is None:
                raise HTTPException(status_code=404, detail="Thread not found.")
            return self.serialize_thread_detail(thread)

    def reset_thread(self, thread_id: str, http_request: Request, user_id: str | None = None) -> dict:
        user_id = _resolve_user_id(http_request, user_id)
        with self.runtime.lock:
            bot = self.runtime.require_bot()
            if bot.conversations.get_thread_for_display(thread_id) is None:
                raise HTTPException(status_code=404, detail="Thread not found.")
            bot.reset_thread(thread_id)
            thread = bot.conversations.get_thread_for_display(thread_id)
            return self.serialize_thread_detail(thread or ThreadState(thread_id=thread_id))

    def chat(self, request: ChatRequest, http_request: Request, user_id: str | None = None) -> dict:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="Message cannot be empty.")

        user_id = _resolve_user_id(http_request, user_id) or None
        if user_id is None and self.config.search.hybrid_search_enabled:
            raise HTTPException(status_code=401, detail="user_id cookie or query param is required.")

        with self.runtime.lock:
            bot = self.runtime.require_bot()
            thread_id = request.thread_id or bot.new_thread(user_id=user_id or "")
            user_profile = resolve_user_profile(
                domain_type=self.config.domain_type,
                user_id=user_id,
            )
            result = bot.chat(
                thread_id=thread_id,
                user_question=message,
                user_id=user_id,
                user_profile=user_profile,
            )
            # result.thread is the live thread object with display_messages already
            # appended — use it directly so we don't lose messages via a second OS load.
            bot.conversations.save_thread(result.thread)
            # After save_thread, display_messages are cleared from memory and live in OS.
            # Reload from OS (or fall back to in-memory) for the API response.
            display_thread = (
                bot.conversations.get_thread_for_display(result.thread_id)
                or ThreadState(thread_id=result.thread_id)
            )
            return {
                "thread": self.serialize_thread_detail(display_thread),
                "answer": result.answer,
            }

    def open_document(self, path: str) -> FileResponse:
        requested_path = self._resolve_document_path(path)

        return FileResponse(
            path=str(requested_path),
            filename=requested_path.name,
            content_disposition_type="inline",
        )

    def view_document(
        self,
        path: str,
        section_index: int | None = None,
        section_title: str = "",
    ) -> HTMLResponse:
        requested_path = self._resolve_document_path(path)
        document_title = self._get_document_title(requested_path)
        open_url = build_document_open_url("", str(requested_path))

        # For HTML files, use a text fragment so the browser scrolls to and
        # highlights the matched section text inside the original document.
        iframe_url = open_url
        if requested_path.suffix.lower() in {FileExtension.HTML, FileExtension.HTM} and section_title.strip():
            from urllib.parse import quote as _quote
            fragment = _quote(section_title.strip()[:120], safe="")
            iframe_url = f"{open_url}#:~:text={fragment}"

        html = self.document_viewer.render(
            document_title=document_title,
            source_path=str(requested_path),
            open_url=open_url,
            iframe_url=iframe_url,
            target_section_title=section_title,
        )
        return HTMLResponse(html)

    @staticmethod
    def build_preview(text: str, limit: int = 96) -> str:
        compact = " ".join(text.split())
        if not compact:
            return ""
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    @staticmethod
    def serialize_message(message) -> dict:
        return {
            "role": message.role,
            "content": message.content,
        }

    @staticmethod
    def serialize_source(source, images: list[str] | None = None) -> dict:
        if hasattr(source, "source_path"):
            title = source.document_title
            path = source.source_path
            original_path = getattr(source, "original_source_path", "") or ""
        else:
            title = source.get("document_title", "")
            path = source.get("source_path", "")
            original_path = source.get("original_source_path", "") or ""
        # Normalise separators so the URL is consistent regardless of how the
        # path was stored (mixed slashes are common on Windows).
        norm_path = path.replace("\\", "/") if path else ""
        # When the source was converted to HTML (e.g. PDF/XLSX → HTML for
        # extraction), open the original file instead of the generated HTML.
        open_path = original_path.replace("\\", "/") if original_path else norm_path
        return {
            "document_title": title,
            "source_path": norm_path,
            "file_name": Path(open_path).name if open_path else "",
            "document_url": build_document_open_url("", open_path) if open_path else "",
            "images": images or [],
        }

    def serialize_thread_summary(self, thread) -> dict:
        preview_source = thread.display_messages[-1].content if thread.display_messages else ""
        return {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "created_at": thread.created_at,
            "updated_at": thread.updated_at,
            "message_count": len(thread.display_messages),
            "preview": self.build_preview(preview_source),
        }

    def serialize_thread_detail(self, thread) -> dict:
        seen_paths: set[str] = set()
        unique_sources = []
        for source in thread.last_answer_sources:
            path = source.source_path if hasattr(source, "source_path") else source.get("source_path", "")
            # Normalize for case-insensitive / separator-agnostic dedup on Windows
            dedup_key = path.lower().replace("\\", "/") if path else ""
            if dedup_key and dedup_key not in seen_paths:
                seen_paths.add(dedup_key)
                unique_sources.append(source)
        return {
            **self.serialize_thread_summary(thread),
            "messages": [self.serialize_message(message) for message in thread.display_messages],
            "sources": [self.serialize_source(s, self._gather_source_images(s)) for s in unique_sources],
            "conversation_summary": thread.conversation_summary,
            "pending_clarification_kind": getattr(thread, "pending_clarification_kind", ""),
            "pending_question": getattr(thread, "pending_question", ""),
            "awaiting_user_input": bool(getattr(thread, "pending_clarification_kind", "")),
        }

    def _gather_source_images(self, source) -> list[str]:
        """Collect images from all in-memory sections belonging to this source document."""
        path = source.source_path if hasattr(source, "source_path") else source.get("source_path", "")
        if not path:
            return []
        norm = path.lower().replace("\\", "/")
        images: list[str] = []
        with self.runtime.lock:
            bot = self.runtime.bot
            if bot is None:
                return []
            for section in bot.corpus.sections.values():
                if section.source_path.lower().replace("\\", "/") == norm:
                    images.extend(section.images)
        return images

    def _resolve_document_path(self, path: str) -> Path:
        requested_path = Path(path).resolve()
        allowed_root = Path(self.config.storage.document_folder).resolve()

        try:
            requested_path.relative_to(allowed_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Document path is outside the policy folder.") from exc

        if not requested_path.is_file():
            raise HTTPException(status_code=404, detail="Document not found.")

        return requested_path

    def _get_document_title(self, requested_path: Path) -> str:
        with self.runtime.lock:
            bot = self.runtime.bot
            if bot is not None:
                for document in bot.documents.values():
                    if Path(document.source_path).resolve() == requested_path:
                        return document.title
        return requested_path.stem.replace("_", " ")
