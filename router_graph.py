# router_graph.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from agents import decide_agent, sub_agents

class AgentState(TypedDict):
    session_id: str
    user_message: str
    chat_history: list
    selected_agent: str
    response: str

def router_node(state: AgentState) -> AgentState:
    selected = decide_agent(state)
    return {**state, "selected_agent": selected}

def sub_agent_node(agent_name: str):
    def run(state: AgentState) -> AgentState:
        output = sub_agents[agent_name](state)
        return {**state, "response": output["response"]}
    return run

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("shopping_assistant", sub_agent_node("shopping_assistant"))
    builder.add_node("similar_products_assistant", sub_agent_node("similar_products_assistant"))

    def route(state: AgentState) -> Literal["shopping_assistant", "similar_products_assistant"]:
        return state["selected_agent"]

    builder.add_conditional_edges("router", route)
    builder.add_edge("shopping_assistant", END)
    builder.add_edge("similar_products_assistant", END)

    return builder.compile()