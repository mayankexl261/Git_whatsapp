import random
import time
import json
import numpy as np
import openai
import streamlit as st


from PIL import Image
import torch
import faiss
import io
from typing import List
from transformers import CLIPProcessor, CLIPModel
import base64
from rapidfuzz import fuzz
import re
import math
from difflib import SequenceMatcher
import ast
import configparser

#importing all the prompts
from prompts import *


config = configparser.ConfigParser()
config.read('config.ini')

print(config)


client = openai.OpenAI(api_key=config['KEY']['open_ai_key'])

#reading vector db and meta data files from config
vector_db = config['SETUP']['vector_db']
index_to_product_json = config['SETUP']['index_to_product_mapping']
metadata_json = config['SETUP']['metadata_mapping']
productID_caption_json = config['SETUP']['productID_caption_mapping']


#reading the vector db
index = faiss.read_index(vector_db)

with open(index_to_product_json, "r", encoding="utf-8") as f:
    index_to_productID = json.load(f)

with open(metadata_json, "r", encoding="utf-8") as f:
    metadata_json = json.load(f)

with open(productID_caption_json, "r", encoding="utf-8") as f:
    productID_caption_json = json.load(f)


def get_openai_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    embedding = response.data[0].embedding
    return embedding


# In[12]:


def query_llm(qa_history: List[dict], user_input: str, temperature=0.5, model="gpt-4o"):
    # qa_history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model=model,
        messages=qa_history,
        temperature=temperature,
    )
    reply = response.choices[0].message.content.strip()
    #qa_history.append({"role": "assistant", "content": reply})
    return reply, qa_history


# In[13]:


def query_llm_new(prompt: str, system_ins: str, temperature=0.5, model="gpt-4o"):
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_ins},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def rephrase_llm(prompt: str, temperature=0.9, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful home and lifestyle shopping assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def rephrase_chats(message_history: list[dict], system_ins: str, temperature=0.5, model="gpt-4o") -> str:
    # Prepend the system message to the conversation
    messages = [{"role": "system", "content": system_ins}] + message_history

    # Send to OpenAI or your client
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    # Return the assistant's response text
    return response.choices[0].message.content.strip()


def extract_json_from_response(response):
    print(f"\n[DEBUG] Input type = {type(response)}")

    # If already parsed as dict, return it
    if isinstance(response, dict):
        print("[DEBUG] Already a dict.")
        return response

    if isinstance(response, str):
        # If wrapped in triple backticks, extract contents
        code_block_pattern = re.compile(r"```(?:json|python)?\s*([\s\S]+?)\s*```", re.MULTILINE)
        match = code_block_pattern.search(response)
        if match:
            response = match.group(1).strip()
        
        return response

    raise TypeError("❌ Model response must be a string or dict.")


def price_order_sorting(filtered_ids, user_input, flag):
    results = []
    for idx, (key, score) in enumerate(filtered_ids.items()): ## change I[0] with filtered_list
        # key = str(int(i))
        # if key in final_next_home_metadata:
        item = metadata_json[key]
        
        # item = filter_dict[item_number]
        price = item.get("price", None)  # assume price field is here
        
        # If no price_filter provided, or price is missing, keep by default
        if price is None:
            results.append((item, float(score)))
        else:
            filter_price = user_input.get("budget")
            comparison = (user_input.get("budget_comparison") or "").lower()
            
            if filter_price is not None:
                # Apply filter based on comparison operator
                if comparison == "greater than":
                    if price > filter_price:
                        results.append((item, float(score)))
                elif comparison == "less than":
                    if price < filter_price:
                        results.append((item, float(score)))
                elif comparison == "equal to":
                    if price == filter_price:
                        results.append((item, float(score)))
            else:
                # Unknown comparison, skip filtering
                results.append((item, float(score)))
        # else:
        #     print(f"Warning: Key {key} not found in final_next_home_metadata")

    # Sort filtered results by total_order descending
    if flag == 1:
        results_sorted = sorted(results, key=lambda x: x[0].get('total_order', 0), reverse=True)[:5]
    else:
        results_sorted = sorted(
            results,
            key=lambda x: (x[1], x[0].get('total_order', 0)),
            reverse=True
        )[:5]

    # print(f"results_sorted: {results_sorted}")

    return results_sorted


# def search_faiss(vector, top_k=100, user_input=None, user_text=None):
#     D, I = index.search(vector, top_k)
#     results = []

#     sim_score = D[0]

#     # # Example data - replace with your own
#     # values = np.array(sim_score)

#     # # 1st derivative
#     # first_diff = np.diff(values)
    
#     # # 2nd derivative
#     # second_diff = np.diff(first_diff)
    
#     # # --- First Derivative Drop-Out Method ---
#     # delta_first_diff = np.diff(first_diff)
#     # first_drop_index = np.argmin(delta_first_diff) + 1  # +1 to align index with original values
    
#     # # --- Max Absolute Second Derivative Method ---
#     # second_deriv_abs_max_index = np.argmax(np.abs(second_diff)) + 2  # +2 for alignment

#     # print(f"\n\nFirst Derivative Drop-Out Elbow Index: {first_drop_index}, Value: {values[first_drop_index]}")
#     # print(f"Max Absolute Second Derivative Elbow Index: {second_deriv_abs_max_index}, Value: {values[second_deriv_abs_max_index]}\n\n")

    
#     # # deriv_index = max(first_drop_index, second_deriv_abs_max_index)
#     # index_candidates = [first_drop_index, second_deriv_abs_max_index]

#     # # Filter candidates less than 10
#     # valid_candidates = [i for i in index_candidates if i < 10]

#     # if valid_candidates:
#     #     deriv_index = max(valid_candidates)
#     # else:
#     #     deriv_index = min(index_candidates)
#     # # deriv_index = second_deriv_abs_max_index

#     prod_names = []

#     # # For example if I is shape (1, top_k):
#     top_k_indices = I[0]  # adjust based on your array shape
#     # # print(f"top_k_indices: {top_k_indices}")
#     filter_indices = []
#     # i = 0

#     # productID_caption_json[prod_id]
#     for idx in top_k_indices:
#         key = str(idx)  # convert integer index to string key
#         if key in index_to_productID: # and sim_score[i] >= values[deriv_index]:
#             prod_id = index_to_productID[key]
#             prod_name = productID_caption_json.get(prod_id, "") #metadata_json[prod_id].get('productName', None)
#             # i += 1
#             # if prod_name:
#             prod_names.append(prod_name)
#             filter_indices.append(prod_id)

#     # print(f"prod_names list: {prod_names}")

#     # user_input = json.load(user_input)
#     # user_text
#     # user_input['prod_desc']
#     # prompt = f"User requirements:\n{user_text}\n\n The following list of product Description:\n{prod_names}"

#     # Create prompt with better formatting
#     prompt = f"User requirements:\n{user_text}\n\nThe following list of product Descriptions:\n"


#     for i, desc in enumerate(prod_names, 1):
#         prompt += f"\n{i}. {desc}\n"

#     print(prompt)

#     product_match_info_json = query_llm_new(prompt, system_ins_topk_description_list)
#     # print(product_match_info_json)

#     # cleaned_product_match_info_json = extract_json_from_response(product_match_info_json)
#     cleaned_product_match_info_json = product_match_info_json.strip('` \n').removeprefix('json').strip()

#     # print(cleaned_product_match_info_json)
#     # cleaned_product_match_info_json = product_match_info_json.strip('` \n').lstrip('json')
#     product_match_info_json = json.loads(cleaned_product_match_info_json)
#     # print(product_match_info_json)

#     print(f"{'Index':<6} {'Flag':<6} {'Sim_score':<6} {'LLM_Score':<7} Reason")
#     print("-" * 50)

#     print('product_match_info_json is printed here!')
#     print(product_match_info_json)

#     print('relevance flag len json')
#     print(len(product_match_info_json['relevance_flags']))

#     for i in range(len(product_match_info_json['relevance_flags'])):
#         flag = product_match_info_json['relevance_flags'][i]
#         score = product_match_info_json['relevance_score'][i]
#         reason = product_match_info_json['relevance_reasons'][i]
#         sim_score_value = sim_score[i]
        
#         print(f"{i+1:<6} {flag:<6} {sim_score_value:<6} {score:<7} {reason}")

#     # ss = bool_list.split(":")[1]

#     # bool_list = ast.literal_eval(ss)

#     # print(dict(zip(prod_names, bool_list)))
#     print('till here in faiss')

#     relevance_flags_list = product_match_info_json['relevance_flags']
#     relevance_flags_score = product_match_info_json['relevance_score']

#     print('still inside')

#     # print(relevance_flags_list)
    
#     filtered_list = [item for item, flag in zip(filter_indices, relevance_flags_list) if flag == 1]
#     # filtered_scores = [score for score, flag in zip(relevance_flags_score, relevance_flags_list) if flag == 1]
#     sim_score_list = [item for item, flag in zip(sim_score, relevance_flags_list) if flag == 1]

#     score_ids_available_dict = dict(zip(filtered_list,sim_score_list))

#     filtered_list_not = [item for item, flag in zip(filter_indices, relevance_flags_list) if flag == 0]
#     # filtered_scores = [score for score, flag in zip(relevance_flags_score, relevance_flags_list) if flag == 1]
#     sim_score_list_not = [item for item, flag in zip(sim_score, relevance_flags_list) if flag == 0]

#     score_ids_not_available_dict = dict(zip(filtered_list_not,sim_score_list_not))

#     # threshold = 50  # 50%

#     # # Keep ids with score > 50
#     # filtered_ids = [id_ for id_, score in zip(filtered_list, filtered_scores) if score > threshold]

#     # print(filtered_list)
#     # print(f"filter_indices: {filter_indices}")


#     print(f'prompt: {prompt}')
#     results_sorted_available = price_order_sorting(score_ids_available_dict, user_input,1)
#     results_sorted_not_available = price_order_sorting(score_ids_not_available_dict, user_input,0)
    
#     return results_sorted_available, results_sorted_not_available


def search_faiss(vector, top_k=100, user_input=None, user_text=None):
    D, I = index.search(vector, top_k)
    results = []
 
    sim_score = D[0]
 
    # # Example data - replace with your own
    # values = np.array(sim_score)
 
    # # 1st derivative
    # first_diff = np.diff(values)
   
    # # 2nd derivative
    # second_diff = np.diff(first_diff)
   
    # # --- First Derivative Drop-Out Method ---
    # delta_first_diff = np.diff(first_diff)
    # first_drop_index = np.argmin(delta_first_diff) + 1  # +1 to align index with original values
   
    # # --- Max Absolute Second Derivative Method ---
    # second_deriv_abs_max_index = np.argmax(np.abs(second_diff)) + 2  # +2 for alignment
 
    # print(f"\n\nFirst Derivative Drop-Out Elbow Index: {first_drop_index}, Value: {values[first_drop_index]}")
    # print(f"Max Absolute Second Derivative Elbow Index: {second_deriv_abs_max_index}, Value: {values[second_deriv_abs_max_index]}\n\n")
 
   
    # # deriv_index = max(first_drop_index, second_deriv_abs_max_index)
    # index_candidates = [first_drop_index, second_deriv_abs_max_index]
 
    # # Filter candidates less than 10
    # valid_candidates = [i for i in index_candidates if i < 10]
 
    # if valid_candidates:
    #     deriv_index = max(valid_candidates)
    # else:
    #     deriv_index = min(index_candidates)
    # # deriv_index = second_deriv_abs_max_index
 
    prod_names = []
 
    # # For example if I is shape (1, top_k):
    top_k_indices = I[0]  # adjust based on your array shape
    # # print(f"top_k_indices: {top_k_indices}")
    filter_indices = []
    # i = 0
 
    # productID_caption_json[prod_id]
    for idx in top_k_indices:
        key = str(idx)  # convert integer index to string key
        if key in index_to_productID: # and sim_score[i] >= values[deriv_index]:
            prod_id = index_to_productID[key]
            prod_name = productID_caption_json.get(prod_id, "") #metadata_json[prod_id].get('productName', None)
            # i += 1
            # if prod_name:
            prod_names.append(prod_name)
            filter_indices.append(prod_id)
 
    # print(f"prod_names list: {prod_names}")
 
    # user_input = json.load(user_input)
    # user_text
    # user_input['prod_desc']
    # prompt = f"User requirements:\n{user_text}\n\n The following list of product Description:\n{prod_names}"
 
    # Create prompt with better formatting
    prompt = f"User requirements:\n{user_text}\n\nThe following list of product Descriptions:\n"
 
 
    for i, desc in enumerate(prod_names, 1):
        prompt += f"\n{i}. {desc}\n"
 
    print(prompt)
 
    product_match_info_json = query_llm_new(prompt, system_ins_topk_description_list)
    # print(product_match_info_json)
 
    # cleaned_product_match_info_json = extract_json_from_response(product_match_info_json)
    cleaned_product_match_info_json = product_match_info_json.strip('` \n').removeprefix('json').strip()
 
    # print(cleaned_product_match_info_json)
    # cleaned_product_match_info_json = product_match_info_json.strip('` \n').lstrip('json')
    product_match_info_json = json.loads(cleaned_product_match_info_json)
    # print(product_match_info_json)
 
    print(f"{'Index':<6} {'Flag':<6} {'Sim_score':<6} {'LLM_Score':<7} {'Prod ID':<7} Reason")
    print("-" * 50)
 
    # Create dictionary with product_id as key and flag + reason as values
    product_info_dict = {}
 
    for i in range(len(product_match_info_json['relevance_flags'])):
        flag = product_match_info_json['relevance_flags'][i]
        score = product_match_info_json['relevance_score'][i]
        reason = product_match_info_json['relevance_reasons'][i]
        sim_score_value = sim_score[i]
        product_id = filter_indices[i]
        product_info_dict[product_id] = {
            'reason': reason,
            'flag': flag
        }
       
        print(f"{i+1:<6} {flag:<6} {sim_score_value:<6} {score:<7} {product_id:<7} {reason}")
 
    # ss = bool_list.split(":")[1]
 
    # bool_list = ast.literal_eval(ss)
 
    # print(dict(zip(prod_names, bool_list)))
 
    relevance_flags_list = product_match_info_json['relevance_flags']
    relevance_score_list = product_match_info_json['relevance_score']
    relevance_reasons_list = product_match_info_json['relevance_reasons']
 
    # print(relevance_flags_list)
   
    filtered_list = [item for item, flag in zip(filter_indices, relevance_flags_list) if flag == 1]
 
    if len(filtered_list) < 6:
 
        follow_up_prompt = f"User requirements:\n{user_text}\n\nThe list of reasons why a product was rejected:\n"
        for i, desc in enumerate(relevance_reasons_list, 1):
            follow_up_prompt += f"\n{i}. {desc}\n"
 
        follow_up_question_on_rejected_prod = query_llm_new(follow_up_prompt, system_ins_restr_not_relevent_prod)
 
        # return None, None, follow_up_question_on_rejected_prod
       
    # filtered_scores = [score for score, flag in zip(relevance_flags_score, relevance_flags_list) if flag == 1]
    sim_score_list = [item for item, flag in zip(sim_score, relevance_flags_list) if flag == 1]
 
    score_ids_available_dict = dict(zip(filtered_list,sim_score_list))
 
    filtered_list_not = [item for item, flag in zip(filter_indices, relevance_flags_list) if flag == 0]
    # filtered_scores = [score for score, flag in zip(relevance_flags_score, relevance_flags_list) if flag == 1]
    sim_score_list_not = [item for item, flag in zip(sim_score, relevance_flags_list) if flag == 0]
 
    score_ids_not_available_dict = dict(zip(filtered_list_not,sim_score_list_not))
 
    # threshold = 50  # 50%
 
    # # Keep ids with score > 50
    # filtered_ids = [id_ for id_, score in zip(filtered_list, filtered_scores) if score > threshold]
 
    # print(filtered_list)
    # print(f"filter_indices: {filter_indices}")
 
 
    # print(prompt)
    results_sorted_available = price_order_sorting(score_ids_available_dict, user_input,1)
    results_sorted_not_available = price_order_sorting(score_ids_not_available_dict, user_input,0)
 
    # results_sorted_available[0].get('key')
    all_key_values = [item[0]['key'] for item in results_sorted_available if 'key' in item[0]]
    print(f"all_key_values: {all_key_values}")
 
    rejected_filtered_product_info = {
        pid: info for pid, info in product_info_dict.items() if pid not in all_key_values
    }
    print(f"rejected_filtered_product_info: {rejected_filtered_product_info}")
   
    if len(filtered_list) < 6:
        return results_sorted_available, results_sorted_not_available, follow_up_question_on_rejected_prod, rejected_filtered_product_info
    else:
        return results_sorted_available, results_sorted_not_available, None, rejected_filtered_product_info


def show_results(results_prod_available, results_prod_not_available):
    st.subheader("Sure! Here are some recommendations for you.")
    
    # Show available products
    num_available = len(results_prod_available)
    if num_available == 0:
        st.warning("No available products found related to your requirement.")
    else:
        # st.markdown("### Available Products")
        cols = st.columns(num_available)
        for col, (meta, score) in zip(cols, results_prod_available):
            img_b64 = meta.get("image_base64")
            prod_desc = meta.get('productName', "Product")
            full_url = meta.get("url", "#")

            if img_b64:
                image_data_url = f"data:image/jpeg;base64, {img_b64}"
                with col:
                    st.markdown(f"""
                        <div style="text-align: left; width: 200px;">
                            <a href="{full_url}" target="_blank">
                                <img src="{image_data_url}" alt="{prod_desc}"
                                    style="width:200px; height:200px; object-fit:cover; border-radius:8px; display: block;"/>
                            </a>
                            <p style="text-align: center; margin-top: 8px;">{prod_desc}<br></p>
                        </div>
                    """, unsafe_allow_html=True)

    # Show not available products
    num_not_available = len(results_prod_not_available)
    if num_not_available > 0:
        st.markdown("### Similar Products")
        cols = st.columns(num_not_available)
        for col, (meta, score) in zip(cols, results_prod_not_available):
            img_b64 = meta.get("image_base64")
            prod_desc = meta.get('productName', "Product")
            full_url = meta.get("url", "#")

            if img_b64:
                image_data_url = f"data:image/jpeg;base64, {img_b64}"
                with col:
                    st.markdown(f"""
                        <div style="text-align: left; width: 200px;">
                            <a href="{full_url}" target="_blank">
                                <img src="{image_data_url}" alt="{prod_desc}"
                                    style="width:200px; height:200px; object-fit:cover; border-radius:8px; display: block;"/>
                            </a>
                            <p style="text-align: center; margin-top: 8px;">{prod_desc}<br></p>
                        </div>
                    """, unsafe_allow_html=True)


def get_rejected_list(reject_list):
    result = {}
    for item_id, data1 in reject_list.items():
        if item_id in metadata_json:
            data2 = metadata_json[item_id]
            flag = data1.get('flag', 0)
            reject_reason = "rejected because lower order count" if flag == 1 else "rejected because llm relevance reason"

            result[item_id] = {
                'reason': data1.get('reason', ''),
                'productName': data2.get('productName', ''),
                'image_base64': data2.get('image_base64', ''),
                'order_value': data2.get('total_order', 0),
                'reject_reason': reject_reason
            }
            
    return result