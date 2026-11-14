from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from app.core.client import chat_client
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.booking import create_booking
from app.services.embeddings import embed_text
from app.services.vectorstore import QdrantStore

SYSTEM_PROMPT = (
    "You are a helpful recruitment assistant. "
    "Use the provided retrieved context to answer questions about ingested documents. "
    "If the answer is not in the context, say you do not know. "
    "When the user wants to book an interview, collect full name, email, date, and time. "
    "Respond professionally and do not use emojis."
)

BOOKING_PROMPT = (
    "Detect interview booking intent from the conversation. "
    "Return ONLY JSON with this shape: "
    '{"is_booking_intent": bool, "full_name": str|null, "email": str|null, '
    '"date": str|null, "time": str|null}. '
    "Use date format YYYY-MM-DD and time format HH:MM (24h). "
    "Use null for unknown fields."
)


class BookingIntent(BaseModel):
    is_booking_intent: bool = False
    full_name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def _chat(messages: list[dict[str, str]], temperature: float | None = None) -> str:
    response = chat_client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=messages,
        temperature=settings.chat_temperature if temperature is None else temperature,
    )
    return (response.choices[0].message.content or "").strip()


def _sanitize_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    allowed_roles = {"user", "assistant", "system"}
    clean: list[dict[str, str]] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in allowed_roles and isinstance(content, str):
            clean.append({"role": role, "content": content})
    return clean


def retrieve_context(query: str) -> str:
    """Search the ingested document knowledge base for the given query."""
    store = QdrantStore()
    store.ensure_collection()
    hits = store.search(embed_text(query), top_k=settings.rag_top_k)
    if not hits:
        return "No relevant context found."
    return "\n\n".join(hit["payload"].get("text", "") for hit in hits)


def book_interview(full_name: str, email: str, date: str, time: str) -> str:
    """Book an interview and return confirmation."""
    db = SessionLocal()
    try:
        booking = create_booking(db, full_name, email, date, time)
    except ValueError as exc:
        return f"Could not book the interview: {exc}. Please correct the details."
    finally:
        db.close()
    return (
        f"Interview confirmed for {full_name} on {date} at {time}. "
        f"Booking reference: #{booking.id}."
    )


class RecruitmentAgent:
    def _extract_booking_intent(
        self, query: str, history: list[dict[str, str]]
    ) -> BookingIntent:
        messages: list[dict[str, str]] = [{"role": "system", "content": BOOKING_PROMPT}]
        messages.extend(_sanitize_history(history)[-10:])
        messages.append({"role": "user", "content": query})
        raw = _chat(messages, temperature=0.0)
        data = _extract_json_object(raw)
        try:
            return BookingIntent.model_validate(data)
        except Exception:
            return BookingIntent()

    @staticmethod
    def _missing_fields(intent: BookingIntent) -> list[str]:
        missing: list[str] = []
        if not intent.full_name:
            missing.append("full name")
        if not intent.email:
            missing.append("email")
        if not intent.date:
            missing.append("date (YYYY-MM-DD)")
        if not intent.time:
            missing.append("time (HH:MM)")
        return missing

    def respond(self, query: str, history: list[dict[str, str]]) -> str:
        booking_intent = self._extract_booking_intent(query, history)
        if booking_intent.is_booking_intent:
            missing = self._missing_fields(booking_intent)
            if missing:
                return "To book your interview, please provide: " + ", ".join(missing) + "."
            return book_interview(
                full_name=booking_intent.full_name or "",
                email=booking_intent.email or "",
                date=booking_intent.date or "",
                time=booking_intent.time or "",
            )

        context = retrieve_context(query)
        if context == "No relevant context found.":
            return "I do not know based on the available document context."

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "Retrieved context:\n"
                    f"{context}\n\n"
                    "Use only this context to answer factual questions."
                ),
            },
        ]
        messages.extend(_sanitize_history(history)[-10:])
        messages.append({"role": "user", "content": query})
        answer = _chat(messages)
        return answer or "I do not know based on the available document context."


@lru_cache
def get_agent() -> RecruitmentAgent:
    return RecruitmentAgent()
