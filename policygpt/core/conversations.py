import uuid

from policygpt.models import ThreadState


class ConversationManager:
    def __init__(self) -> None:
        self.threads: dict[str, ThreadState] = {}

    def new_thread(self) -> str:
        thread_id = str(uuid.uuid4())
        self.threads[thread_id] = ThreadState(thread_id=thread_id)
        return thread_id

    def reset_thread(self, thread_id: str) -> None:
        self.threads[thread_id] = ThreadState(thread_id=thread_id)

    def get_thread(self, thread_id: str) -> ThreadState:
        if thread_id not in self.threads:
            self.threads[thread_id] = ThreadState(thread_id=thread_id)
        return self.threads[thread_id]

    def list_threads(self) -> list[ThreadState]:
        return sorted(self.threads.values(), key=lambda thread: thread.updated_at, reverse=True)
