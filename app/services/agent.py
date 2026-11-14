from __future__ import annotations

from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from app.core.config import settings
from app.db.session import SessionLocal
from app.services.booking import create_booking
from app.services.embeddings import embed_text
from app.services.vectorstore import QdrantStore

SYSTEM_PROMPT = (
    "You are a helpful recruitment assistant. "
    "Use the retrieve_context tool to answer questions about ingested documents, "
    "and answer only from retrieved context. If the answer is not in context, "
    "say you do not know. When the user wants to book an interview, make sure "
    "you have their full name, email, date and time, then call book_interview "
    "to confirm it. Respond professionally and do not use emojis."
)


def _resolved_chat_credentials() -> tuple[str, str, str | None]:
    endpoint = settings.azure_openai_chat_endpoint or settings.azure_openai_endpoint
    api_key = (
        settings.azure_openai_chat_api_key.get_secret_value()
        if settings.azure_openai_chat_api_key
        else settings.azure_openai_api_key.get_secret_value()
    )
    api_version = settings.azure_openai_chat_api_version or settings.openai_api_version
    return endpoint, api_key, api_version


def _normalize_endpoint(endpoint: str) -> str:
    cleaned_endpoint = endpoint.rstrip("/")
    marker = "/openai/v1"
    marker_index = cleaned_endpoint.lower().find(marker)
    if marker_index != -1:
        cleaned_endpoint = cleaned_endpoint[:marker_index]
    return cleaned_endpoint


@lru_cache
def _build_model() -> ChatOpenAI:
    endpoint, api_key, api_version = _resolved_chat_credentials()
    normalized_endpoint = _normalize_endpoint(endpoint)
    host = (urlparse(normalized_endpoint).hostname or "").lower()

    kwargs: dict[str, Any] = {
        "model": settings.azure_openai_chat_deployment,
        "api_key": api_key,
        "base_url": f"{normalized_endpoint}/openai/v1/",
        "temperature": settings.chat_temperature,
    }
    if api_version and not host.endswith("services.ai.azure.com"):
        kwargs["default_query"] = {"api-version": api_version}

    return ChatOpenAI(**kwargs)


@tool
def retrieve_context(query: str) -> str:
    """Search the ingested document knowledge base for the given query."""
    clean_query = query.strip()
    if not clean_query:
        return "No relevant context found."

    store = QdrantStore()
    store.ensure_collection()
    hits = store.search(embed_text(clean_query), top_k=settings.rag_top_k)
    if not hits:
        return "No relevant context found."
    return "\n\n".join(hit["payload"].get("text", "") for hit in hits)


@tool
def book_interview(full_name: str, email: str, date: str, time: str) -> str:
    """Book an interview.

    Requires full name, email, date (YYYY-MM-DD), and time (HH:MM).
    """
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


@lru_cache
def _build_graph() -> CompiledStateGraph:
    return create_react_agent(
        _build_model(),
        tools=[retrieve_context, book_interview],
        prompt=SYSTEM_PROMPT,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
            continue

        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

    return "\n".join(parts).strip()


class RecruitmentAgent:
    def __init__(self) -> None:
        self._graph = _build_graph()

    @staticmethod
    def _sanitize_history(history: list[dict[str, str]]) -> list[tuple[str, str]]:
        clean: list[tuple[str, str]] = []
        for item in history[-10:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                clean.append((role, content))
        return clean

    def respond(self, query: str, history: list[dict[str, str]]) -> str:
        clean_query = query.strip()
        if not clean_query:
            return "Please share your question."

        conversation = self._sanitize_history(history)
        conversation.append(("user", clean_query))

        state = self._graph.invoke({"messages": conversation})
        messages = state.get("messages", []) if isinstance(state, dict) else []

        for message in reversed(messages):
            if isinstance(message, BaseMessage) and message.type == "ai":
                text = _content_to_text(message.content)
                if text:
                    return text
            if isinstance(message, dict) and message.get("role") == "assistant":
                text = _content_to_text(message.get("content"))
                if text:
                    return text

        return "I do not know based on the available document context."


@lru_cache
def get_agent() -> RecruitmentAgent:
    return RecruitmentAgent()
