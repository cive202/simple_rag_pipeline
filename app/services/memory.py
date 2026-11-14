import json

import redis

from app.core.config import settings

HISTORY_TTL_SECONDS = settings.session_ttl_seconds
MAX_MESSAGES = 20

_client = redis.from_url(settings.redis_url, decode_responses=True)


class ConversationMemory:
    """Redis-backed store for multi-turn conversation history."""

    @staticmethod
    def _key(user_id: str) -> str:
        return f"chat:{user_id}"

    def load(self, user_id: str) -> list[dict[str, str]]:
        raw = _client.get(self._key(user_id))
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        messages: list[dict[str, str]] = []
        for item in data:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append(
                    {"role": str(item["role"]), "content": str(item["content"])}
                )
        return messages

    def append(self, user_id: str, role: str, content: str) -> None:
        messages = self.load(user_id)
        messages.append({"role": role, "content": content})
        _client.set(
            self._key(user_id),
            json.dumps(messages[-MAX_MESSAGES:]),
            ex=HISTORY_TTL_SECONDS,
        )
