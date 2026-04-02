from policygpt import Config, PolicyGPTBot, create_ready_bot
from policygpt.cli import PolicyGPTCli

__all__ = ["Config", "PolicyGPTBot", "create_ready_bot"]


def main() -> None:
    PolicyGPTCli(config=Config.from_env()).run()


if __name__ == "__main__":
    main()
