from __future__ import annotations

from collections import deque

from .models import CrossModalMessage


class SharedMessageBus:
    """Small in-process abstract bus that only stores summarized messages."""

    def __init__(self, max_history: int = 32) -> None:
        self._messages: deque[CrossModalMessage] = deque(maxlen=max_history)

    def publish(self, message: CrossModalMessage) -> None:
        self._messages.append(message)

    def recent(self) -> list[CrossModalMessage]:
        return list(self._messages)

    def recent_for_tick(self, timestamp: float) -> list[CrossModalMessage]:
        return [message for message in self._messages if message.timestamp == timestamp]

    def clear_tick(self, timestamp: float) -> None:
        kept = [message for message in self._messages if message.timestamp != timestamp]
        self._messages.clear()
        self._messages.extend(kept)
