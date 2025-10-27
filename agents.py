# agents.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Dict



# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Master agent prompt for routing
MASTER_AGENT_PROMPT = """You are a master AI agent designed to route user queries to the correct tool. You have access to the following tools: 
1. "shopping_assistant": helps find and buy home/lifestyle products. 
2. "similar_products_assistant": recommends products similar or complementary to an existing item. 
Based on the user's query, respond ONLY with the tool name to be used.

Examples: 
User: "I need a new blue sofa" 
Response: "shopping_assistant"

User: "Show me cushions that go well with a blue sofa" 
Response: "similar_products_assistant"

User's query: {user_message}
Response:"""

router_prompt = PromptTemplate(
    input_variables=["user_message"],
    template=MASTER_AGENT_PROMPT
)

router_chain = LLMChain(llm=llm, prompt=router_prompt)

def decide_agent(payload: dict) -> str:
    return router_chain.run({"user_message": payload["user_message"]}).strip()

# Shopping assistant with RAG + FAISS
def shopping_assistant_agent(payload: dict) -> dict:
    user_message = payload["user_message"]

    # Get embedding vector
    query_embedding = embedding_model.embed_query(user_message)

    # Search FAISS index
    top_k = 5
    search_results = search_faiss(query_embedding, top_k)

    if not search_results:
        response_text = "Sorry, I couldn't find any matching products."
    else:
        response_text = "Here are some products I found:\n"
        for i, item in enumerate(search_results, 1):
            title = item.get("title", "Unknown product")
            desc = item.get("description", "")
            response_text += f"{i}. {title} - {desc}\n"

    return {
        "agent": "shopping_assistant",
        "response": response_text
    }

# Similar products assistant - dummy placeholder
def similar_products_agent(payload: dict) -> dict:
    user_message = payload["user_message"]
    return {
        "agent": "similar_products_assistant",
        "response": f"Here are items that go well with: {user_message}"
    }

# Register sub agents
sub_agents = {
    "shopping_assistant": shopping_assistant_agent,
    "similar_products_assistant": similar_products_agent
}