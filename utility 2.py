import random
import time
import json
import numpy as np
import openai
import streamlit as st
import pandas as pd


from PIL import Image
# import torch
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

from numpy.linalg import norm

#importing all the prompts
from prompts import *

from sentence_transformers import SentenceTransformer

import concurrent.futures

from concurrent.futures import ThreadPoolExecutor, as_completed


config = configparser.ConfigParser()
config.read('config.ini')

print(config)
debug= True

def printf(text= "", flag= False):
    if flag: 
        print(text)

client = openai.OpenAI(api_key=config['KEY']['open_ai_key'])
model = SentenceTransformer('all-MiniLM-L6-v2')

#reading vector db and meta data files from config
vector_db = config['SETUP']['vector_db']
index_to_product_json = config['SETUP']['index_to_product_mapping']
metadata_json = config['SETUP']['metadata_mapping']
productID_caption_json = config['SETUP']['productID_caption_mapping']


#reading the vector db
# t0= time.time()
index = faiss.read_index(vector_db)
# time_taken= time.time() - t0
# printf(f"time_taken: {time_taken}")

with open(index_to_product_json, "r", encoding="utf-8") as f:
    index_to_productID = json.load(f)

with open(metadata_json, "r", encoding="utf-8") as f:
    metadata_json = json.load(f)

with open(productID_caption_json, "r", encoding="utf-8") as f:
    productID_caption_json = json.load(f)

with open("items_to_category.json", "r", encoding="utf-8") as f:
    items_to_category = json.load(f)

with open("sample_testing_output_new.json", "r", encoding="utf-8") as f:
    product_json = json.load(f)

df = pd.read_csv("20k_metadata_embeddings_new.csv")
df_categories = pd.read_csv('captioning_csv.csv') 

metadata_keys = [col for col in df.columns if col != "product_id"]
def safe_literal_eval(x):
    if isinstance(x, str) and x.strip():  # Check non-empty string
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None
    elif pd.isna(x) or x == "":
        return None
    return x

for key in metadata_keys:
    df[key] = df[key].apply(safe_literal_eval)

product_20k_index = faiss.read_index("20k_product_name_faiss_new.index")

# Load ID mapping
with open("20k_id_mapping_new.json", "r") as f:
    mapping = json.load(f)
int_to_str_id = {int(k): v for k, v in mapping["int_to_str_id"].items()}
str_to_int_id = {v: k for k, v in int_to_str_id.items()}


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

def cosine_sim(a, b):
    if a is None or b is None:
        return None
    a = np.array(a)
    b = np.array(b)
    if norm(a) == 0 or norm(b) == 0:
        return 0
    return np.dot(a, b) / (norm(a) * norm(b))

def process_product(pid, product_data, user_text):
    if pid not in product_data:
        return f"[SKIP] ID {pid} not found."

    item = product_data[pid]
    json_str = json.dumps(item, ensure_ascii=False)
    stripped = json_str[1:-1]  # remove outer braces

    prompt = f"User requirements:\n{user_text}\n\nProduct Descriptions:\n\n. {stripped}\n"
    
    relevence_json = query_llm_new(prompt, system_ins_topk_description_list_new, temperature=0.2)

    relevence_json = relevence_json.strip('` \n').lstrip('json')
    relevence_result = json.loads(relevence_json)
    # relevence_result

    # 🔧 Simulate your LLM call here
    # You can replace this with OpenAI API or other model inference
    # response = openai.ChatCompletion.create(...) or similar
    # print(f"[PROCESSING] ID {pid}")
    # return f"[DONE] ID {pid}: {stripped[:80]}..."  # Print first 80 chars

    return pid, relevence_result

def filter_products_by_metadata(best_category, user_input, top_n=20):

    prompt = user_input['product_name']

    user_requirement = df_categories[df_categories['category'].astype(str).str.strip()==best_category]['User Requirement'].astype(str).str.strip().iloc[0]

    system_ins = f"""
        You will receive a product descriptions, along with its category: {best_category}.
        
        Your task is to extract details ({user_requirement}) from description and return them strictly as a valid JSON object, without any explanations or additional text.
        
        Important: Include the Category, Brand, Product name, and a Product summary along with extracted details. The Product summary should be a concise, natural-language description suitable for vector embedding, incorporating all extracted information from the other fields. Ensure that no detail explicitly stated in the descriptions is omitted.
        
        Only extract information directly mentioned in the descriptions. Do not infer or add any details not explicitly provided. Use UK English and maintain a neutral, factual, and informative tone in the product_summary.
        
        Return only a JSON object formatted as follows:
        {{
          "category": string,
          "product_name": string,
          "brand": string or null,
          ...
          "product_summary": string
        }}

        Note: All values for each key should be string or null only, no need to include list or dict in any value in json. Category, Product name and Product summary can not be None or empty
        """
    
    final_caption = query_llm_new(prompt, system_ins)

    final_caption = final_caption.strip('` \n').lstrip('json')
    final_caption = json.loads(final_caption)
    # combined_caption

    # Embed user input fields (all keys present in user_json)
    user_embeds = { key: model.encode(final_caption[key]).tolist() for key in metadata_keys if key in final_caption and final_caption[key] is not None}

    
    excluded_keys = {"product_name", "product_summary"}
    
    weights = {k: 1.0 for k in final_caption.keys() if k not in excluded_keys}

    # Initialize similarity score
    df['similarity_score'] = 0.0

    # Weights for metadata fields (adjust as needed)
    # weights = {
    #     "category": 1.0,
    #     "brand": 1.0,
    #     "colour": 1.0,
    #     "material": 1.0,
    #     "pattern": 1.0,
    #     "border": 1.0,
    #     "shape": 1.0
    # }

    # Calculate weighted similarity for each metadata field
    for key, user_vec in user_embeds.items():
        sims = []
        for emb in df[key]:
            # emb = ast.literal_eval(emb)
            sim = cosine_sim(user_vec, emb)
            if sim is None:
                sim = 0
            sims.append(sim)
        df['similarity_score'] += np.array(sims) * weights.get(key, 1.0)

    # Get top N product_ids by similarity score
    top_products = df.nlargest(top_n, 'similarity_score')['product_id'].tolist()
    sims_score_prod = df.nlargest(top_n, 'similarity_score')['similarity_score'].tolist()
    print("Filtered products by metadata similarity:", top_products)

    # Remove keys with None values
    cleaned_user_text = {k: v for k, v in final_caption.items() if v is not None}

    user_text = cleaned_user_text
    # Convert to JSON string
    user_text = json.dumps(user_text, ensure_ascii=False)

    # Strip the outer braces
    user_text = user_text[1:-1]

    # process_product(top_products, product_json, user_text)

    # 🔁 Multithreading setup
    MAX_THREADS = 20  # adjust based on your API rate limits or CPU

    result = {}

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_product, pid, product_json, user_text) for pid in top_products]

        for future in as_completed(futures):
            prod_id, result_dict = future.result()
            result[prod_id] = result_dict

    product_match_info_json = {
        "relevance_flags": [],
        "relevance_score": [],
        "relevance_reasons": []
    }

    filter_indices = top_products
    results_dict = result
    sim_score = sims_score_prod
    for prod_id in filter_indices:
        res = results_dict.get(prod_id)
        if res:
            product_match_info_json["relevance_flags"].append(res["relevance_flags"])
            product_match_info_json["relevance_score"].append(res["relevance_score"])
            product_match_info_json["relevance_reasons"].append(res["relevance_reasons"])
        else:
            # Handle missing case if any
            product_match_info_json["relevance_flags"].append(0)
            product_match_info_json["relevance_score"].append(0)
            product_match_info_json["relevance_reasons"].append("No result")


    
 
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
 
    if len(filtered_list) < 5:
 
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
    # print(f"results_sorted_available length: {results_sorted_available}")
    results_sorted_not_available = price_order_sorting(score_ids_not_available_dict, user_input,0)
 
    # results_sorted_available[0].get('key')
    all_key_values = [item[0]['key'] for item in results_sorted_available if 'key' in item[0]]
    # print(f"all_key_values: {all_key_values}")
 
    rejected_filtered_product_info = {
        pid: info for pid, info in product_info_dict.items() if pid not in all_key_values
    }
    # print(f"rejected_filtered_product_info: {rejected_filtered_product_info}")
   
    if len(filtered_list) < 5:
        return results_sorted_available, results_sorted_not_available, follow_up_question_on_rejected_prod, rejected_filtered_product_info
    else:
        return results_sorted_available, results_sorted_not_available, None, rejected_filtered_product_info

    # return top_products, sims_score_prod


from collections import Counter

def search_most_common_category_through_product_name(user_query, top_k=10):
    # Embed user summary
    query_vec = model.encode(user_query).astype('float32')
    faiss.normalize_L2(query_vec.reshape(1, -1))

    # Map filtered product string IDs to FAISS int IDs
    # filtered_int_ids = set(str_to_int_id[pid] for pid in filtered_product_ids)

    # Search FAISS index with a larger k to filter later
    D, I = product_20k_index.search(query_vec.reshape(1, -1), top_k)

    # print(I[0])

    
    # # Filter results to keep only those in filtered_int_ids
    # filtered_results = [(dist, idx) for dist, idx in zip(D[0], I[0]) if idx in filtered_int_ids]
    # filtered_results.sort(key=lambda x: x[0], reverse=True)

    # # Take top_k after filtering
    # filtered_results = filtered_results[:top_k]

    # # Map back to original string product IDs
    final_results = [int_to_str_id[idx] for idx in I[0] if idx in int_to_str_id]

    # Map str_ids to categories, ignoring any str_id not in dict
    categories = [items_to_category[str_id] for str_id in final_results if str_id in items_to_category]

    # Count frequency of each category
    counter = Counter(categories)
    
    # Get the most common category and its count
    most_common_category, count = counter.most_common(1)[0]
    
    print(f"Most frequent category: {most_common_category} (occurs {count} times)")

    return most_common_category

def extract_json_from_response(response):
    printf(f"\n[DEBUG] Input type = {type(response)}", flag= debug)

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
        results_sorted = sorted(results, key=lambda x: x[0].get('total_order', 0), reverse=True)[:8]
    else:
        results_sorted = sorted(
            results,
            key=lambda x: (x[1], x[0].get('total_order', 0)),
            reverse=True
        )[:8]

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

def create_prompt(user_text, product_desc):
    prompt = f"User requirements:\n{user_text}\n\nProduct Descriptions:\n\n1. {product_desc}\n"
    return prompt

def call_model_single_product(user_text, product_desc, prod_id):
    prompt = create_prompt(user_text, product_desc)
    result = query_llm_new(prompt, system_ins_topk_description_list, 0.1)
    result = result.strip('` \n').removeprefix('json').strip()
    result = json.loads(result)
    # Return prod_id with result for identification
    return (prod_id, result)

def search_faiss(vector, top_k=100, user_input=None, user_text=None):
    start_time  = time.time()
    D, I = index.search(vector, top_k)
    print(f"vector search time : {time.time()-start_time}")
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
    
    # start_time  = time.time()
    # Create prompt with better formatting
    # prompt = f"User requirements:\n{user_text}\n\nThe following list of product Descriptions:\n"
 
 
    # for i, desc in enumerate(prod_names, 1):
    #     prompt += f"\n{i}. {desc}\n"
 
    # print(prompt)
 
    # product_match_info_json = query_llm_new(prompt, system_ins_topk_description_list, 0.1)
    # print(product_match_info_json)
 
    # cleaned_product_match_info_json = extract_json_from_response(product_match_info_json)
    # cleaned_product_match_info_json = product_match_info_json.strip('` \n').removeprefix('json').strip()
 
    # print(cleaned_product_match_info_json)
    # cleaned_product_match_info_json = product_match_info_json.strip('` \n').lstrip('json')
    # product_match_info_json = json.loads(cleaned_product_match_info_json)
    # print(f"final filter search time : {time.time()-start_time}")
    # print(product_match_info_json)

    start_time  = time.time()
    max_parallel_calls = 20
    results_dict = {}

    # Assuming filter_indices and prod_names are same length and aligned
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_calls) as executor:
        futures = [
            executor.submit(call_model_single_product, user_text, desc, prod_id)
            for prod_id, desc in zip(filter_indices, prod_names)
        ]

        for future in concurrent.futures.as_completed(futures):
            prod_id, result = future.result()
            # print(f"prod_id : {prod_id}\n")
            # print(f"result : {result}\n")
            results_dict[prod_id] = result

    # print(f"final_results_dict : {results_dict}")

    print(f"final filter search time : {time.time()-start_time}")

    product_match_info_json = {
        "relevance_flags": [],
        "relevance_score": [],
        "relevance_reasons": []
    }

    for prod_id in filter_indices:
        res = results_dict.get(prod_id)
        if res:
            product_match_info_json["relevance_flags"].append(res["relevance_flags"])
            product_match_info_json["relevance_score"].append(res["relevance_score"])
            product_match_info_json["relevance_reasons"].append(res["relevance_reasons"])
        else:
            # Handle missing case if any
            product_match_info_json["relevance_flags"].append(0)
            product_match_info_json["relevance_score"].append(0)
            product_match_info_json["relevance_reasons"].append("No result")


    
 
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
 
    if len(filtered_list) < 5:
 
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
    # print(f"results_sorted_available length: {results_sorted_available}")
    results_sorted_not_available = price_order_sorting(score_ids_not_available_dict, user_input,0)
 
    # results_sorted_available[0].get('key')
    all_key_values = [item[0]['key'] for item in results_sorted_available if 'key' in item[0]]
    # print(f"all_key_values: {all_key_values}")
 
    rejected_filtered_product_info = {
        pid: info for pid, info in product_info_dict.items() if pid not in all_key_values
    }
    # print(f"rejected_filtered_product_info: {rejected_filtered_product_info}")
   
    if len(filtered_list) < 5:
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