from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from policygpt.config import Config
from policygpt.core.document_links import build_document_open_url, build_document_view_url
from policygpt.api.renderers.document_viewer import DocumentViewerRenderer
from policygpt.api.runtime import ServerRuntime
from policygpt.api.renderers.ui import WebUIRenderer
from policygpt.extraction.file_extractor import FileExtractor


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
        self.extractor = FileExtractor(config)

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
        app.add_api_route("/api/documents/open", self.open_document, methods=["GET"], response_class=FileResponse)
        app.add_api_route("/api/documents/view", self.view_document, methods=["GET"], response_class=HTMLResponse)
        return app

    def index(self) -> HTMLResponse:
        return HTMLResponse(self.ui_renderer.render_index())

    def domain_ui(self) -> dict:
        profile = self.config.domain_profile
        return {
            "assistant_label": profile.ui_assistant_label,
            "eyebrow": profile.ui_eyebrow,
            "description": profile.ui_description,
            "prompt_chips": [
                {"label": label, "prompt": prompt}
                for label, prompt in profile.ui_prompt_chips
            ],
        }

    def health(self) -> dict:
        with self.runtime.lock:
            bot = self.runtime.bot
            return {
                "status": self.runtime.status,
                "error": self.runtime.error,
                "document_folder": self.runtime.document_folder,
                "document_count": self.runtime.document_count,
                "section_count": self.runtime.section_count,
                "thread_count": len(bot.threads) if bot else 0,
                "progress": self.runtime.progress_payload(),
            }

    def usage(self) -> dict:
        with self.runtime.lock:
            return self.runtime.usage_payload()

    def list_threads(self) -> dict:
        with self.runtime.lock:
            bot = self.runtime.require_bot()
            return {
                "items": [self.serialize_thread_summary(thread) for thread in bot.list_threads()],
            }

    def create_thread(self) -> dict:
        with self.runtime.lock:
            bot = self.runtime.require_bot()
            thread_id = bot.new_thread()
            return self.serialize_thread_detail(bot.get_thread(thread_id))

    def get_thread(self, thread_id: str) -> dict:
        with self.runtime.lock:
            bot = self.runtime.require_bot()
            thread = bot.threads.get(thread_id)
            if thread is None:
                raise HTTPException(status_code=404, detail="Thread not found.")
            return self.serialize_thread_detail(thread)

    def reset_thread(self, thread_id: str) -> dict:
        with self.runtime.lock:
            bot = self.runtime.require_bot()
            if thread_id not in bot.threads:
                raise HTTPException(status_code=404, detail="Thread not found.")
            bot.reset_thread(thread_id)
            return self.serialize_thread_detail(bot.get_thread(thread_id))

    def chat(self, request: ChatRequest, http_request: Request) -> dict:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="Message cannot be empty.")

        # Extract user_id from cookie — mandatory when hybrid search is enabled.
        user_id = http_request.cookies.get("user_id")
        if user_id is None and self.config.hybrid_search_enabled:
            raise HTTPException(status_code=401, detail="user_id cookie is required.")

        with self.runtime.lock:
            bot = self.runtime.require_bot()
            thread_id = request.thread_id or bot.new_thread()
            result = bot.chat(thread_id=thread_id, user_question=message, user_id=user_id)
            thread = bot.get_thread(result.thread_id)
            return {
                "thread": self.serialize_thread_detail(thread),
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
        document_title, sections = self._load_document_sections(requested_path)
        normalized_sections = [
            {
                "dom_id": f"section-{index}",
                "title": item["title"],
                "text": item["text"],
                "order_index": item["order_index"],
            }
            for index, item in enumerate(sections)
        ]

        target_dom_id = normalized_sections[0]["dom_id"] if normalized_sections else ""
        target_title = normalized_sections[0]["title"] if normalized_sections else ""

        if normalized_sections:
            matched_section = None
            if section_index is not None:
                matched_section = next(
                    (item for item in normalized_sections if item["order_index"] == section_index),
                    None,
                )
            if matched_section is None and section_title.strip():
                normalized_title = section_title.strip().casefold()
                matched_section = next(
                    (item for item in normalized_sections if str(item["title"]).strip().casefold() == normalized_title),
                    None,
                )
            if matched_section is not None:
                target_dom_id = str(matched_section["dom_id"])
                target_title = str(matched_section["title"])

        html = self.document_viewer.render(
            document_title=document_title,
            source_path=str(requested_path),
            sections=normalized_sections,
            target_section_id=target_dom_id,
            target_section_title=target_title or section_title or "Matched section",
            open_url=build_document_open_url("", str(requested_path)),
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
    def serialize_source(source) -> dict:
        return {
            "document_title": source.document_title,
            "section_title": source.section_title,
            "section_order_index": source.section_order_index,
            "source_path": source.source_path,
            "file_name": Path(source.source_path).name,
            "score": round(source.score, 4),
            "document_url": build_document_view_url(
                "",
                source_path=source.source_path,
                section_index=source.section_order_index,
                section_title=source.section_title,
            ),
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
        return {
            **self.serialize_thread_summary(thread),
            "messages": [self.serialize_message(message) for message in thread.display_messages],
            "sources": [self.serialize_source(source) for source in thread.last_answer_sources],
            "conversation_summary": thread.conversation_summary,
        }

    def _resolve_document_path(self, path: str) -> Path:
        requested_path = Path(path).resolve()
        allowed_root = Path(self.config.document_folder).resolve()

        try:
            requested_path.relative_to(allowed_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Document path is outside the policy folder.") from exc

        if not requested_path.is_file():
            raise HTTPException(status_code=404, detail="Document not found.")

        return requested_path

    def _load_document_sections(self, requested_path: Path) -> tuple[str, list[dict[str, object]]]:
        with self.runtime.lock:
            bot = self.runtime.bot
            if bot is not None:
                for document in bot.documents.values():
                    if Path(document.source_path).resolve() != requested_path:
                        continue
                    return (
                        document.title,
                        [
                            {
                                "title": section.title,
                                "text": section.raw_text,
                                "order_index": section.order_index,
                            }
                            for section in document.sections
                        ],
                    )

        title, extracted_sections = self.extractor.extract(str(requested_path))
        sections = [
            {
                "title": section_title,
                "text": section_text,
                "order_index": index,
            }
            for index, (section_title, section_text) in enumerate(extracted_sections)
        ]
        return title, sections
