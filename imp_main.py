import os
from typing import TypedDict, List, Optional, Literal

from fastapi import FastAPI, Form, Response
from xml.sax.saxutils import escape as xml_escape
from twilio.rest import Client as TwilioClient
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from fastapi.responses import ORJSONResponse
import numpy as np
import json

# Load environment variables from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import configparser

#importing the utility functions
from utility import *
from prompts_old import *

#reading the config
config = configparser.ConfigParser()
config.read('config.ini')

print(config)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


origins = [
    "http://localhost",
    "http://localhost:5173",  # If your frontend runs on React dev server
]


# Initialize OpenAI LLM and Embeddings
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize Twilio client if credentials are available
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g. "whatsapp:+1415..."

twilio_client = None
if TWILIO_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = TwilioClient(TWILIO_SID, TWILIO_AUTH_TOKEN)
        logger.info("Twilio client initialized")
        # quick smoke test (non-fatal) if desired; commented out to avoid extra API call
        # logger.info(twilio_client.api.accounts.list(limit=1))
    except Exception as e:
        logger.exception("Failed to initialize Twilio client: %s", e)
        twilio_client = None
else:
    logger.info("Twilio credentials not found in environment; Twilio integration disabled")


# Define typed state for LangGraph
class ChatState(TypedDict, total=False):
    session_id: str
    user_message: str
    chat_history: List[dict]
    selected_agent: Optional[Literal["shopping_assistant", "style_assistant", "consultation_agent"]]
    response: Optional[dict]

# Master agent system prompt for routing
system_prompt = """
You are a master AI agent responsible for routing user queries to the correct assistant. Available tools: 
1. shopping_assistant — Handles all detailed product-related queries, especially those explicitly about finding, buying, or purchasing home and lifestyle products. Any user request that involves acquiring or shopping for a product should be sent here.
2. consultation_agent — Helps with other queries, like clarifying vague requests, asking follow-up questions, or providing general assistance. 
3. style_assistant — Handles queries related to matching, complementing, or finding similar products to existing items (e.g., "Show me sofas that go well with a green rug", "I want curtains that look good with blue walls", or "Find something similar to this chair").

Your job: 
- If the query is clearly about shopping or buying a product, choose 'shopping_assistant'. 
- If the query talks about products that go well with, match, complement, or are similar to an existing product, choose 'style_assistant'. 
- If the query is vague, unclear, or needs clarification but seems related to lifestyle or shopping, choose 'consultation_agent'.

Respond ONLY with one of:
- shopping_assistant
- consultation_agent
- style_assistant

Examples:
User: I need a new yoga mat
Response: shopping_assistant

User: Can you help me decide what kind of furniture to buy?
Response: consultation_agent

User: Show me sofas that go well with a green rug
Response: style_assistant
"""



sessions = {}

# Router node: master agent decides which sub-agent to route to
def router_node(state: ChatState) -> ChatState:
    chat_history = state.get("chat_history", [])
    user_message = state.get("user_message", "")
    
    messages = [SystemMessage(content=system_prompt)]
    
    for turn in chat_history:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            # Assistant messages should not be HumanMessage; use AIMessage if available
            messages.append(AIMessage(content=content))
    
    # Add current user message
    messages.append(HumanMessage(content=user_message))
    
    tool_name = llm(messages).content.strip().lower()
    print(tool_name)
    
    valid_tools = {"shopping_assistant", "style_assistant", "consultation_agent"}
    
    if tool_name not in valid_tools:
        # If unexpected output, fallback to consultation_agent or generic response
        # return {**state, "response": {"assistant_message": "Please ask a relevant query."}}
        tool_name = "consultation_agent"

    
    return {**state, "selected_agent": tool_name}

# Shopping assistant node: search vector DB and return products
def shopping_assistant_node(request: ChatState) -> ChatState:
    print("✅ [DEBUG] Entered REAL shopping_assistant_node")
    print(f"✅ [DEBUG] Entered with {request}")
    session_id = request['session_id']
    user_message = request['user_message']
    existing_chat = request.get("chat_history", [])

    # chat_history_list = sessions.get(session_id, [{"role": "system", "content": convo_system_ins}])
    chat_history_list = existing_chat + [{"role": "system", "content": convo_system_ins}]
    chat_history_list.append({"role":"user", "content":user_message})

    print(chat_history_list)

    assistant_message = ""
    results_sorted_available = []
    rejected_items = {}

    if len(chat_history_list) == 2:
        query_type = query_llm_new(user_message, SYSTEM_INS_QUERY_TYPE_PROMPT)

        if query_type == "detailed":
            print(chat_history_list)
            full_convo = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in chat_history_list
                if msg["role"] in ["user", "assistant"]
                )
        
            print(f'full_convo: {full_convo}')
            json_string = query_llm_new(full_convo, system_ins_str_restructure)
            print(f'json_string: {json_string}')
            
            try:
                cleaned_string = json_string.strip('` \n').lstrip('json')
                filters_dict = json.loads(cleaned_string)
                print(filters_dict)
                
            except json.JSONDecodeError:
                print("Error: Could not parse the AI response. Please try again.")
                filters_dict = {}
            
            compact_text = filters_dict["product_summary"] 
            print(f'compact_text product_summary: {compact_text}')

            compact_text = query_llm_new(compact_text, system_ins_description)
            print(f'compact_text system_ins_description for embedding: {compact_text}')


            embedding_vector = get_openai_embedding(compact_text)
            embedding_vector = np.array(embedding_vector).astype("float32")
            embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)      
            
            print(embedding_vector)
            # Ensure 2D shape for FAISS
            if embedding_vector.ndim == 1:
                embedding_vector = embedding_vector.reshape(1, -1)
            
            results_sorted_available, results_sorted_not_available, follow_up_question_on_rejected_prod, rejected_filtered_product_info = search_faiss(embedding_vector, 20, filters_dict, compact_text)
            print('rejected_filtered_product_info')

            rejected_items = get_rejected_list(rejected_filtered_product_info)
            # print(rejected_items)
            if follow_up_question_on_rejected_prod is None:
                assistant_message = query_llm_new(user_message, SYSTEM_INS_ASST_RESPONSE)

            if follow_up_question_on_rejected_prod is not None:
                # final_message = "Sorry, we don't have enough products as per your prefrence."
                assistant_message = f"{follow_up_question_on_rejected_prod}"
        else:
            print('else')
            assistant_message = query_llm_new(user_message, SYSTEM_INS_ASST_RESPONSE)
    
    else:
        print(chat_history_list)
        full_convo = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history_list
            if msg["role"] in ["user", "assistant"]
            )
        print(f'full_convo: {full_convo}')
        json_string = query_llm_new(full_convo, system_ins_str_restructure)
        print(f'json_string: {json_string}')
        
        try:
            cleaned_string = json_string.strip('` \n').lstrip('json')
            filters_dict = json.loads(cleaned_string)
            print(filters_dict)
                
        except json.JSONDecodeError:
            print("Error: Could not parse the AI response. Please try again.")
            filters_dict = {}
            
        compact_text = filters_dict["product_summary"] 
        print(f'compact_text product_summary: {compact_text}')

        compact_text = query_llm_new(compact_text, system_ins_description)
        print(f'compact_text system_ins_description for embedding: {compact_text}')

            
        embedding_vector = get_openai_embedding(compact_text)
        embedding_vector = np.array(embedding_vector).astype("float32")
        embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)      
            
        print(embedding_vector)
        # Ensure 2D shape for FAISS
        if embedding_vector.ndim == 1:
            embedding_vector = embedding_vector.reshape(1, -1)
            
        results_sorted_available, results_sorted_not_available, follow_up_question_on_rejected_prod, rejected_filtered_product_info = search_faiss(embedding_vector, 20, filters_dict, compact_text)
        print('rejected_filtered_product_info')
        rejected_items = get_rejected_list(rejected_filtered_product_info)

        print('above assit')

        if follow_up_question_on_rejected_prod is None:
            assistant_message, chat_list = query_llm(chat_history_list, user_message)

        if follow_up_question_on_rejected_prod is not None:
                # final_message = "Sorry, we don't have enough products as per your prefrence."
                assistant_message = f"{follow_up_question_on_rejected_prod}"

    
    chat_history_list.append({"role":"assistant", "content":assistant_message})
    sessions[session_id] = chat_history_list

    print('before res')

    res = {
        "session_id": session_id,
        "assistant_message" : assistant_message,
        "search_results": results_sorted_available,
        "updated_chat_history": chat_history_list,
        "rejected_items": rejected_items
    }

    return {
        **request,
        "response": res
    }

# Goto products assistant node: reframe vague queries
def goto_products_assistant_node(state: ChatState) -> ChatState:
    print("✅ [DEBUG] Entered goto_products_assistant_node")
    user_message = state["user_message"]
    session_id = state["session_id"]
    chat_history = state.get("chat_history", [])

    reframed = f"Can you help me find options for: {user_message}?"
    assistant_message = query_llm_new(reframed, SYSTEM_INS_ASST_RESPONSE)

    updated_chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

    return {
        **state,
        "response": {
            "session_id": session_id,
            "assistant_message": assistant_message,
            "updated_chat_history": updated_chat_history,
            "recommended_products": []
        }
    }

def style_assistant_node(state: ChatState) -> ChatState:
    print("✅ [DEBUG] Entered styling_assistant_node")
    print(f"✅ [DEBUG] Entered with {state}")
    user_message = state["user_message"]
    session_id = state["session_id"]
    chat_history = state.get("chat_history", [])

    full_conversation = chat_history + [{"role": "user", "content": user_message}]
    assistant_message = rephrase_chats(full_conversation, STYLE_ASSISTANT_PROMPT)
    print(assistant_message)

    shopping_state = {**state, "user_message": assistant_message }

    shopping_response_state = shopping_assistant_node(shopping_state)
    print('shopping_response_state')
    print(shopping_response_state)
    return shopping_response_state


def consultation_agent_node(state: ChatState) -> ChatState:
    print("✅ [DEBUG] Entered consultation_agent_node")
    print(f"✅ [DEBUG] Entered with {state}")
    user_message = state["user_message"]
    session_id = state["session_id"]
    chat_history = state.get("chat_history", [])

    full_conversation = chat_history + [{"role": "user", "content": user_message}]

    assistant_message = rephrase_chats(full_conversation, CONSULTATION_AGENT_INS)
    updated_chat_history = full_conversation + [{"role": "assistant", "content": assistant_message}]

    return {
        **state,
        "response": {
            "session_id": session_id,
            "assistant_message": assistant_message,
            "updated_chat_history": updated_chat_history,
            "recommended_products": []
        }
    }

# Build the LangGraph with nodes and edges
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("router", router_node)
    graph.add_node("shopping_assistant", shopping_assistant_node)
    graph.add_node("style_assistant", style_assistant_node)
    graph.add_node("consultation_agent", consultation_agent_node)


    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state.get("selected_agent", "done"),
        {
            "shopping_assistant": "shopping_assistant",
            "style_assistant": "style_assistant",
            "consultation_agent": "consultation_agent",
            "done": END
        }
    )

    graph.add_edge("shopping_assistant", END)
    graph.add_edge("style_assistant", END)
    graph.add_edge("consultation_agent", END)


    return graph.compile()

graph_app = build_graph()

# FastAPI app and request model
app = FastAPI(default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],    # You can limit this to specific methods like ["GET", "POST"]
    allow_headers=["*"],    # You can limit this to specific headers
)

class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    chat_history: List[dict]

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    state: ChatState = {
        "session_id": request.session_id,
        "user_message": request.user_message,
        "chat_history": request.chat_history
    }

    print('state')
    print(state)
    

    result = graph_app.invoke(state)
    print('we have result here')
    print(result)
    return result["response"]


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    """Handle incoming WhatsApp webhook (form-encoded) and return TwiML XML.

    This endpoint responds with XML (TwiML) as required by WhatsApp webhook integrations
    like Twilio. The TwiML will contain a single <Message> with the assistant's reply.
    """

    # Use the WhatsApp sender phone as session id (strip the leading "whatsapp:" if present)
    session_id = From.replace("whatsapp:", "") if From else "unknown"

    existing_chat = sessions.get(session_id, [])

    state: ChatState = {
        "session_id": session_id,
        "user_message": Body,
        "chat_history": existing_chat
    }

    result = graph_app.invoke(state)

    resp = result.get("response", {}) if isinstance(result, dict) else {}
    print(resp)

    # Try to get assistant message from the graph response
    assistant_text = None
    if isinstance(resp, dict):
        assistant_text = resp.get("assistant_message") or resp.get("assistantResponse")

    if not assistant_text:
        assistant_text = "Sorry, I couldn't process your request right now."

    # Update sessions store with returned chat history if present
    try:
        updated_chat = resp.get("updated_chat_history") or resp.get("updated_chat_history")
        if isinstance(updated_chat, list):
            sessions[session_id] = updated_chat
    except Exception:
        pass

    # Build a product list section if search_results are present
    product_section = ""
    try:
        search_results = resp.get("search_results") if isinstance(resp, dict) else None
        if search_results and isinstance(search_results, list) and len(search_results) > 0:
            lines = ["Top product matches:"]
            # search_results is a list of tuples (meta, score)
            for i, item in enumerate(search_results[:10], start=1):
                try:
                    meta = item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else item
                    name = meta.get('productName') or meta.get('title') or meta.get('name') or meta.get('product_name') or "Product"
                    desc = meta.get('description') or meta.get('short_description') or meta.get('productDescription') or meta.get('product_desc') or ""
                    price = meta.get('price')
                    url = meta.get('url') or meta.get('product_url') or meta.get('shopping_link') or ""
                    # Try to get canonical product id for building image link
                    prod_id = meta.get('key') or meta.get('productId') or meta.get('product_id') or meta.get('id') or None
                    # total_order (popularity / order count)
                    total_order = meta.get('total_order') or meta.get('totalOrder') or meta.get('order_value') or meta.get('totalOrders') or None
                    image_url = ""
                    if prod_id:
                        # Build image URL using the provided template
                        try:
                            image_url = f"https://xcdn.next.co.uk/common/items/default/default/itemimages/3_4Ratio/product/lge/{prod_id}s.jpg?im=Resize,width=400"
                        except Exception:
                            image_url = ""

                    line = f"{i}. {name}"
                    # include short description inline with product name
                    if desc:
                        short_desc = (desc[:240] + '...') if len(desc) > 240 else desc
                        line += f" - {short_desc}"
                    # brand and category (on their own lines)
                    category = meta.get('productCategory') or meta.get('category') or meta.get('department') or ""
                    brand = meta.get('brandName') or meta.get('brand') or meta.get('manufacturer') or ""
                    if brand:
                        line += f"\nBrand: {brand}"
                    if category:
                        line += f"\nCategory: {category}"
                    if price is not None:
                        line += f"\nPrice: {price}"
                    if total_order is not None:
                        line += f"\nTotal orders: {total_order}"
                    if url:
                        line += f"\nLink: {url}"
                    if image_url:
                        line += f"\nImage: {image_url}"

                    lines.append(line)
                except Exception:
                    continue

            product_section = "\n\n" + "\n\n".join(lines)
            # capture the first product image (if any) to include as Twilio media
            first_image = None
            try:
                first_item = search_results[0]
                first_meta = first_item[0] if isinstance(first_item, (list, tuple)) and len(first_item) > 0 else first_item
                first_prod_id = first_meta.get('key') or first_meta.get('productId') or first_meta.get('product_id') or first_meta.get('id')
                if first_prod_id:
                    first_image = f"https://xcdn.next.co.uk/common/items/default/default/itemimages/3_4Ratio/product/lge/{first_prod_id}s.jpg?im=Resize,width=400"
            except Exception:
                first_image = None
    except Exception:
        product_section = ""

    # Compose final message: assistant message followed by product list (if any)
    formatted_message = f"{assistant_text}{product_section}"

    # Escape text for XML safety
    safe_text = xml_escape(str(formatted_message))

    # If a Twilio client and a valid Twilio WhatsApp sender are configured, send the message via Twilio
    sent_via_twilio = False
    twilio_error = None
    if twilio_client and TWILIO_WHATSAPP_FROM:
        try:
            # Ensure 'to' has the whatsapp: prefix
            to_number = From if (From and From.startswith("whatsapp:")) else f"whatsapp:{session_id}"
            # If we have a first_image, send it as media via Twilio (media_url expects a list)
            if 'first_image' in locals() and first_image:
                message = twilio_client.messages.create(
                    body=str(formatted_message),
                    from_=TWILIO_WHATSAPP_FROM,
                    to=to_number,
                    media_url=[first_image]
                )
            else:
                message = twilio_client.messages.create(
                    body=str(formatted_message),
                    from_=TWILIO_WHATSAPP_FROM,
                    to=to_number
                )
            sent_via_twilio = True
            logger.info("Sent message via Twilio to %s, SID=%s", to_number, getattr(message, 'sid', None))
        except Exception as e:
            twilio_error = str(e)
            logger.exception("Failed to send WhatsApp message via Twilio: %s", e)

    # Build TwiML response (useful if Twilio expects it); if we sent via REST API we still return XML but include status
    status_note = ""
    if sent_via_twilio:
        status_note = " (sent via Twilio)"
    elif twilio_error:
        status_note = f" (Twilio send failed: {xml_escape(twilio_error)})"

    twiml = f"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Message>{safe_text}{xml_escape(status_note)}</Message></Response>"

    return Response(content=twiml, media_type="application/xml")


@app.get("/hello")
def say_hello():
    return {"message": "Hello, welcome to FastAPI!"}
