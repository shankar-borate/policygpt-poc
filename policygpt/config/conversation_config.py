"""Conversation / context window configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConversationConfig:
    max_recent_messages: int = 6
    recent_chat_message_char_limit: int = 0
    summarize_after_turns: int = 8
