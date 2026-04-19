from policygpt.config import Config
from policygpt.factory import create_ready_bot


class PolicyGPTCli:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()

    def run(self) -> None:
        folder = self.config.storage.document_folder
        print(f"Ingesting folder: {folder}")
        bot = create_ready_bot(folder=folder, config=self.config)

        current_thread = bot.new_thread()
        print(f"\nReady. Current thread: {current_thread}")
        self.print_help()

        while True:
            try:
                user_input = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue

            if user_input == "/exit":
                break

            if user_input == "/new":
                current_thread = bot.new_thread()
                print(f"Switched to new thread: {current_thread}")
                continue

            if user_input == "/reset":
                bot.reset_thread(current_thread)
                print(f"Thread reset: {current_thread}")
                continue

            if user_input == "/threads":
                print("Threads:")
                for thread_id in bot.threads.keys():
                    marker = " <current>" if thread_id == current_thread else ""
                    print(f"  {thread_id}{marker}")
                continue

            if user_input.startswith("/use "):
                _, thread_id = user_input.split(" ", 1)
                current_thread = thread_id.strip()
                bot.get_thread(current_thread)
                print(f"Switched to thread: {current_thread}")
                continue

            if user_input == "/sources":
                thread = bot.get_thread(current_thread)
                if not thread.last_answer_sources:
                    print("No sources yet.")
                else:
                    print("Last answer sources:")
                    for source in thread.last_answer_sources:
                        print(
                            "  -",
                            f"{source.document_title} :: {source.section_title} ({source.source_path})",
                        )
                continue

            answer = bot.ask(current_thread, user_input)
            print(f"\nBot> {answer}")

    @staticmethod
    def print_help() -> None:
        print("""
Commands:
  /new               -> create a new thread and switch to it
  /reset             -> reset current thread memory
  /threads           -> list thread ids
  /use <thread_id>   -> switch to an existing thread
  /sources           -> show last answer sources
  /exit              -> quit

Anything else is treated as a user question.
""")
