from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from policygpt.config import Config
from policygpt.server.runtime import ServerRuntime
from policygpt.server.ui import WebUIRenderer


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None


class PolicyApiServer:
    def __init__(self, config: Config, web_dir: Path) -> None:
        self.config = config
        self.web_dir = web_dir
        self.runtime = ServerRuntime(config)
        self.ui_renderer = WebUIRenderer(web_dir)

    def build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_: FastAPI):
            self.runtime.start_indexing()
            yield

        app = FastAPI(title="Policy GPT", lifespan=lifespan)
        app.mount("/static", StaticFiles(directory=str(self.web_dir)), name="static")

        app.add_api_route("/", self.index, methods=["GET"], response_class=HTMLResponse)
        app.add_api_route("/api/health", self.health, methods=["GET"])
        app.add_api_route("/api/usage", self.usage, methods=["GET"])
        app.add_api_route("/api/threads", self.list_threads, methods=["GET"])
        app.add_api_route("/api/threads", self.create_thread, methods=["POST"])
        app.add_api_route("/api/threads/{thread_id}", self.get_thread, methods=["GET"])
        app.add_api_route("/api/threads/{thread_id}/reset", self.reset_thread, methods=["POST"])
        app.add_api_route("/api/chat", self.chat, methods=["POST"])
        app.add_api_route("/api/documents/open", self.open_document, methods=["GET"], response_class=FileResponse)
        return app

    def index(self) -> HTMLResponse:
        return HTMLResponse(self.ui_renderer.render_index())

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

    def chat(self, request: ChatRequest) -> dict:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="Message cannot be empty.")

        with self.runtime.lock:
            bot = self.runtime.require_bot()
            thread_id = request.thread_id or bot.new_thread()
            result = bot.chat(thread_id=thread_id, user_question=message)
            thread = bot.get_thread(result.thread_id)
            return {
                "thread": self.serialize_thread_detail(thread),
                "answer": result.answer,
            }

    def open_document(self, path: str) -> FileResponse:
        requested_path = Path(path).resolve()
        allowed_root = Path(self.config.document_folder).resolve()

        try:
            requested_path.relative_to(allowed_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Document path is outside the policy folder.") from exc

        if not requested_path.is_file():
            raise HTTPException(status_code=404, detail="Document not found.")

        return FileResponse(
            path=str(requested_path),
            filename=requested_path.name,
            content_disposition_type="inline",
        )

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
            "source_path": source.source_path,
            "file_name": Path(source.source_path).name,
            "score": round(source.score, 4),
            "document_url": f"/api/documents/open?path={quote(source.source_path, safe='')}",
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
