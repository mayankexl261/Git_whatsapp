import os
from typing import TypedDict, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# -----------------------------
# Load .env file (OpenAI Key)
# -----------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -----------------------------
# LangChain LLM
# -----------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# -----------------------------
# Define State Schema (TypedDict)
# -----------------------------
class ChatState(TypedDict, total=False):
    session_id: str
    user_message: str
    chat_history: List[dict]
    selected_agent: Optional[Literal["shopping_assistant", "goto_products_assistant"]]
    response: Optional[str]

# -----------------------------
# System Prompt for Router
# -----------------------------
system_prompt = (
    "You are a master AI agent responsible for routing user queries to the correct assistant.\n\n"
    "Available tools:\n"
    "1. shopping_assistant - Helps users find and buy home/lifestyle products.\n"
    "2. goto_products_assistant - Reframes vague or generic product-related queries and redirects them to shopping_assistant.\n\n"
    "Your task:\n"
    "- Analyze the latest user message and chat history.\n"
    "- Determine the most suitable assistant.\n"
    "- Respond ONLY with one of the following tool names:\n"
    "    - shopping_assistant\n"
    "    - goto_products_assistant\n"
    "- Do NOT include any other text or explanation.\n\n"
    "Examples:\n"
    "User: I’m looking for new bedsheets.\n"
    "Response: shopping_assistant\n\n"
    "User: I want to see some product options.\n"
    "Response: goto_products_assistant\n\n"
    "Now, based on the user message and chat history, respond with the correct tool name."
)

# -----------------------------
# Master Agent (Router Node)
# -----------------------------
def router_node(state: ChatState) -> ChatState:
    chat_history = state.get("chat_history", [])
    user_message = state.get("user_message", "")

    messages = [SystemMessage(content=system_prompt)]

    # Add chat history
    for turn in chat_history:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(HumanMessage(content=f"(assistant): {content}"))

    # Add latest user message
    messages.append(HumanMessage(content=user_message))

    # LLM decides tool
    tool_name = llm(messages).content.strip().lower()

    # Fallback if invalid tool
    if tool_name not in {"shopping_assistant", "goto_products_assistant"}:
        return {**state, "response": "Please ask a relevant query."}

    return {**state, "selected_agent": tool_name}

# -----------------------------
# Sub-Agent: Shopping Assistant
# -----------------------------
def shopping_assistant_node(state: ChatState) -> ChatState:
    user_message = state["user_message"]
    response = f"[shopping_assistant] Handling your request: '{user_message}'"
    return {**state, "response": response}

# -----------------------------
# Sub-Agent: Goto Products Assistant
# -----------------------------
def goto_products_assistant_node(state: ChatState) -> ChatState:
    user_message = state["user_message"]
    reframed = f"Can you help me find options for: {user_message}?"
    response = f"[goto_products_assistant] Reframed: {reframed}"
    return {**state, "response": response}

# -----------------------------
# Build LangGraph
# -----------------------------
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("router", router_node)
    graph.add_node("shopping_assistant", shopping_assistant_node)
    graph.add_node("goto_products_assistant", goto_products_assistant_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state.get("selected_agent", "done"),
        {
            "shopping_assistant": "shopping_assistant",
            "goto_products_assistant": "goto_products_assistant",
            "done": END
        }
    )

    graph.add_edge("shopping_assistant", END)
    graph.add_edge("goto_products_assistant", END)

    return graph.compile()

graph_app = build_graph()

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    chat_history: List[dict]

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    state = {
        "session_id": request.session_id,
        "user_message": request.user_message,
        "chat_history": request.chat_history,
    }

    result = graph_app.invoke(state)
    return {"response": result["response"]}