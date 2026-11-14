from app.services.agent import get_agent
from app.services.memory import ConversationMemory


def run_chat_turn(user_id: str, query: str) -> str:
    memory = ConversationMemory()
    history = memory.load(user_id)
    response = get_agent().respond(query=query, history=history)
    memory.append(user_id, "user", query)
    memory.append(user_id, "assistant", response)
    return response
