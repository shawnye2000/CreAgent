import cv2
import base64
import json
import logging
from typing import Any, Dict, Optional, List, Tuple
import re
import itertools
import random
from ..llm import *
from yacs.config import CfgNode
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import math
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# logger
def set_logger(log_file, name="default"):
    """
    Set logger.
    Args:
        log_file (str): log file path
        name (str): logger name
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the 'log' folder if it doesn't exist
    log_folder = os.path.join(output_folder, "log")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Create the 'message' folder if it doesn't exist
    message_folder = os.path.join(output_folder, "message")
    if not os.path.exists(message_folder):
        os.makedirs(message_folder)
    log_file = os.path.join(log_folder, log_file)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


# json
def load_json(json_file: str, encoding: str = "utf-8") -> Dict:
    with open(json_file, "r", encoding=encoding) as fi:
        data = json.load(fi)
    return data


def save_json(
    json_file: str,
    obj: Any,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: Optional[int] = None,
    **kwargs,
) -> None:
    with open(json_file, "w", encoding=encoding) as fo:
        json.dump(obj, fo, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def bytes_to_json(data: bytes) -> Dict:
    return json.loads(data)


def dict_to_json(data: Dict) -> str:
    return json.dumps(data)


# cfg
def load_cfg(cfg_file: str, new_allowed: bool = True) -> CfgNode:
    """
    Load config from file.
    Args:
        cfg_file (str): config file path
        new_allowed (bool): whether to allow new keys in config
    """
    with open(cfg_file, "r") as fi:
        cfg = CfgNode.load_cfg(fi)
    cfg.set_new_allowed(new_allowed)
    return cfg


def add_variable_to_config(cfg: CfgNode, name: str, value: Any) -> CfgNode:
    """
    Add variable to config.
    Args:
        cfg (CfgNode): config
        name (str): variable name
        value (Any): variable value
    """
    cfg.defrost()
    cfg[name] = value
    cfg.freeze()
    return cfg


def merge_cfg_from_list(cfg: CfgNode, cfg_list: list) -> CfgNode:
    """
    Merge config from list.
    Args:
        cfg (CfgNode): config
        cfg_list (list): a list of config, it should be a list like
        `["key1", "value1", "key2", "value2"]`
    """
    cfg.defrost()
    cfg.merge_from_list(cfg_list)
    cfg.freeze()
    return cfg


def extract_item_names(observation: str, action: str = "RECOMMENDER") -> List[str]:
    """
    Extract item names from observation
    Args:
        observation: observation from the environment
        action: action type, RECOMMENDER or SOCIAL
    """
    item_names = []
    if observation.find("<") != -1:
        matches = re.findall(r"<(.*?)>", observation)
        item_names = []
        for match in matches:
            item_names.append(match)
    elif observation.find(";") != -1:
        item_names = observation.split(";")
        item_names = [item.strip(" '\"") for item in item_names]
    elif action == "RECOMMENDER":
        matches = re.findall(r'"([^"]+)"', observation)
        for match in matches:
            item_names.append(match)
    elif action == "SOCIAL":
        matches = re.findall(r'[<"]([^<>"]+)[">]', observation)
        for match in matches:
            item_names.append(match)
    return item_names


def layout_img(background, img, place: Tuple[int, int]):
    """
    Place the image on a specific position on the background
    Args:
        background: background image
        img: the specified image
        place: [top, left]
    """
    back_h, back_w, _ = background.shape
    height, width, _ = img.shape
    for i, j in itertools.product(range(height), range(width)):
        if img[i, j, 3]:
            background[place[0] + i, place[1] + j] = img[i, j, :3]


def get_avatar1(idx):
    """
    Retrieve the avatar for the specified index and encode it as a byte stream suitable for display in a text box.
    Args:
        idx (int): The index of the avatar, used to determine the path to the avatar image.
    """
    img = cv2.imread(f"./asset/img/v_1/{idx}.png")
    base64_str = cv2.imencode(".png", img)[1].tostring()
    avatar = "data:image/png;base64," + base64.b64encode(base64_str).decode("utf-8")
    msg = f'<img src="{avatar}" style="width: 100%; height: 100%; margin-right: 50px;">'
    return msg


def get_avatar2(idx):
    """
    Retrieve the avatar for the specified index and encode it as a Base64 data URI.
    Args:
        idx (int): The index of the avatar, used to determine the path to the avatar image.
    """
    img = cv2.imread(f"./asset/img/v_1/{idx}.png")
    base64_str = cv2.imencode(".png", img)[1].tostring()
    return "data:image/png;base64," + base64.b64encode(base64_str).decode("utf-8")


def html_format(orig_content: str):
    """
    Convert the original content to HTML format.
    Args:
        orig_content (str): The original content.
    """
    new_content = orig_content.replace("<", "")
    new_content = new_content.replace(">", "")
    for name in [
        "Eve",
        "Tommie",
        "Jake",
        "Lily",
        "Alice",
        "Sophia",
        "Rachel",
        "Lei",
        "Max",
        "Emma",
        "Ella",
        "Sen",
        "James",
        "Ben",
        "Isabella",
        "Mia",
        "Henry",
        "Charlotte",
        "Olivia",
        "Michael",
    ]:
        html_span = "<span style='color: red;'>" + name + "</span>"
        new_content = new_content.replace(name, html_span)
    new_content = new_content.replace("['", '<span style="color: #06A279;">[\'')
    new_content = new_content.replace("']", "']</span>")
    return new_content


# border: 0;
def chat_format(msg: Dict):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """
    html_text = "<br>"
    avatar = get_avatar2(msg["agent_id"])
    html_text += (
        f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    )
    html_text += f'<img src="{avatar}" style="width: 10%; height: 10%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;">'
    html_text += f'<div style="background-color: #FAE1D1; color: black; padding: 10px; border-radius: 10px; max-width: 80%;">'
    html_text += f'{msg["content"]}'
    html_text += f"</div></div>"
    return html_text


def rec_format(msg: Dict):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """
    html_text = "<br>"
    avatar = get_avatar2(msg["agent_id"])
    html_text += (
        f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    )
    html_text += f'<img src="{avatar}" style="width: 10%; height: 10%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;">'
    html_text += f'<div style="background-color: #D9E8F5; color: black; padding: 10px; border-radius: 10px; max-width: 80%;">'
    html_text += f'{msg["content"]}'
    html_text += f"</div></div>"
    return html_text


def social_format(msg: Dict):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """
    html_text = "<br>"
    avatar = get_avatar2(msg["agent_id"])
    html_text += (
        f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    )
    html_text += f'<img src="{avatar}" style="width: 10%; height: 10%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;">'
    html_text += f'<div style="background-color: #DFEED5; color: black; padding: 10px; border-radius: 10px; max-width: 80%;">'
    html_text += f'{msg["content"]}'
    html_text += f"</div></div>"
    return html_text


def round_format(round: int, agent_name: str):
    """
    Convert the round information to HTML format.
    Args:
        round (int): The round number.
        agent_name (str): The agent name.
    """
    round_info = ""
    round_info += f'<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 20px; color: #000000; font-weight: bold;">'
    round_info += f"&nbsp;&nbsp; Round: {round}  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Actor: {agent_name}  &nbsp;&nbsp;"
    round_info += f"</div>"
    return round_info


def ensure_dir(dir_path):
    """
    Make sure the directory exists, if it does not exist, create it
    Args:
        dir_path (str): The directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_id(dir_name):
    ensure_dir(dir_name)
    existed_id = set()
    for f in os.listdir(dir_name):
        existed_id.add(f.split("-")[0])
    id = random.randint(1, 999999999)
    while id in existed_id:
        id = random.randint(1, 999999999)
    return id


def get_llm(config, logger, api_key):
    """
    Get the large language model.
    Args:
        config (CfgNode): The config.
        logger (Logger): The logger.
        api_key (str): The API key.
    """
    if config["llm"] == "gpt-4":
        LLM = ChatOpenAI(
            max_tokens=config["max_token"],
            temperature=config["temperature"],
            openai_api_key=api_key,
            model="gpt-4",
            max_retries=config["max_retries"]
        )
    elif config["llm"] == "gpt-3.5-16k":
        LLM = ChatOpenAI(
            max_tokens=config["max_token"],
            temperature=config["temperature"],
            openai_api_key=api_key,
            model="gpt-3.5-turbo-16k",
            max_retries=config["max_retries"]
        )
    elif config["llm"] == "gpt-3.5":
        LLM = ChatOpenAI(
            max_tokens=config["max_token"],
            temperature=config["temperature"],
            openai_api_key=api_key,
            model="gpt-3.5-turbo",
            max_retries=config["max_retries"]
        )
    elif config["llm"] == "custom":
        LLM = CustomLLM(max_token=2048, logger=logger)
    else:
        raise ValueError(f"Invalid llm: {config['llm']}")
    return LLM


def is_chatting(agent, agent2):
    """Determine if agent1 and agent2 is chatting"""
    name = agent.name
    agent_name2 = agent2.name
    return (
        (agent2.event.target_agent)
        and (agent.event.target_agent)
        and (name in agent2.event.target_agent)
        and (agent_name2 in agent.event.target_agent)
    )

def get_feature_description(feature):
    """Get description of given features."""
    descriptions = {
        "Watcher": "Choose movies, enjoy watching, and provide feedback and ratings to the recommendation system.",
        "Explorer": "Search for movies heard of before and expand movie experiences.",
        "Critic": "Demanding high standards for movies and the recommendation system, may criticize both the recommendation system and the movies.",
        "Chatter": "Engage in private conversations, trust friends' recommendations.",
        "Poster": "Enjoy publicly posting on social media and sharing content and insights with more people."
    }
    features = feature.split(";")
    descriptions_list = [descriptions[feature] for feature in features if feature in descriptions]
    return ".".join(descriptions_list)

def count_files_in_directory(target_directory:str):
    """Count the number of files in the target directory"""
    return len(os.listdir(target_directory))

def get_avatar_url(id:int,gender:str,type:str="origin",role=False):
    if role:
        target='/asset/img/avatar/role/'+gender+'/'
        return target+str(id%10)+'.png'
    target='/asset/img/avatar/'+type+"/"+gender+'/'
    return target+str(id%10)+'.png'


def calculate_entropy(movie_types):
    type_freq = {}
    for movie_type in movie_types:
        if movie_type in type_freq:
            type_freq[movie_type] += 1
        else:
            type_freq[movie_type] = 1

    total_movies = len(movie_types)

    entropy = 0
    for key in type_freq:
        prob = type_freq[key] / total_movies
        entropy -= prob * math.log2(prob)

    return entropy


def get_entropy(inters, data):
    genres = data.get_genres_by_id(inters)
    entropy = calculate_entropy(genres)
    return entropy


def get_embedding_model():
    model_name = '/home/xiaopeng_ye/LLMs/all-mpnet-base-v2' #'/new_disk2/xiaopeng_ye/experiment/Agent4Fairness/qwen/Qwen2-7B-Instruct' #"sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    # openai_api_key = "EMPTY"
    # openai_api_base = "http://localhost:8000/v1"
    # embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key,
    #                                     openai_api_base=openai_api_base)
    embeddings_size =  768 # 1536 #
    return embeddings_size, embeddings_model

def standardize(numbers):
    mean = sum(numbers) / len(numbers)
    std_dev = (sum((x - mean) ** 2 for x in numbers) / len(numbers)) ** 0.5
    standardized_numbers = [(x - mean) / std_dev for x in numbers]
    return standardized_numbers


def extract_provider_name(self, messages):
    provider_names = [provider['name'] for provider in self.simulator.data.providers.values()]
    # print(f'provider names:{provider_names}')
    for provider_name in provider_names:
        if messages.find(provider_name) != -1:
            return provider_name

    raise ValueError('Provider Not Found.')

def response_to_item(response, choose_genre):
    result = response
    import ast
    # print(f'---response:{result} ---')
    start_index = result.find("{")
    end_index = result.find("}")
    if end_index == -1:
        dict_result = result[start_index:]
    else:
        dict_result = result[start_index: end_index + 1]
    # print(f'---{dict_result} ---')
    # result = re.findall(dict_pattern, result)
    # print(f'---222{result} ---')

    try:
        eval_result = ast.literal_eval(dict_result)
        key_value = list(eval_result.keys())
    except:
        try:
            import json_repair
            eval_result = json_repair.repair_json(dict_result, return_objects=True)
            key_value = list(eval_result.keys())
        except:
            result = correct_SyntaxError(result)
            start_index = result.find("{")
            end_index = result.find("}")
            dict_result = result[start_index: end_index + 1]
            eval_result = ast.literal_eval(dict_result)
            key_value = list(eval_result.keys())
    # print(f'---eval_result{eval_result} ---')


    try:
        item_name = eval_result['name']
        item_genre = eval_result['genre']
        item_tags = eval_result['tags']
        item_description = eval_result['description']
    except:
        try:
            item_name = eval_result[key_value[0]]
        except:
            item_name = ''
        try:
            item_genre = eval_result[key_value[1]]
        except:
            item_genre = choose_genre
        try:
            item_tags = eval_result[key_value[2]]
        except:
            item_tags = []
        try:
            item_description = eval_result[key_value[3]]
        except:
            item_description = ''
    genre_list = get_item_categories()
    if item_genre not in genre_list:
        # match_item_genre = find_first_match(item_genre, genre_list)
        # if match_item_genre != None:
        #     return item_name, match_item_genre,item_tags, item_description
        # else:
        print(f'not match item_genre:{item_genre}')

        # match_item_genre = find_most_similar_embedding(item_genre, genre_list)

        return item_name, choose_genre, item_tags, item_description
    else:
        return item_name, item_genre, item_tags, item_description




    # 加载预训练的BERT模型和分词器


def embed_text(text):
    model_name = '/home/xiaopeng_ye/LLMs/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def find_most_similar_embedding(target, strings_list):
    target_embedding = embed_text(target)
    embeddings = [embed_text(text) for text in strings_list]

    # 计算余弦相似度
    similarities = cosine_similarity(target_embedding, torch.cat(embeddings))
    most_similar_index = similarities.argmax()

    return strings_list[most_similar_index]


def find_first_match(text, words_list):
    pattern = '|'.join(re.escape(word) for word in words_list)
    match = re.search(pattern, text)
    return match.group(0) if match else None




def correct_SyntaxError(result):
    from openai import OpenAI
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    messages = [
        {"role": "user",
         "content": ("Correct its JSON syntax errors and convert it into the correct JSON dictionary format:"
                    '''\n {"name": "item_name", "genre": "genre_name", "tags": [tag1, tag2, tag3], "description": "item_description_text"}''' 
                     f'The sentence that need to be corrected are as follows: {result}')
         },
    ]

    # print(f'Error correct:{messages}')
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="llama3-8b",
        messages=messages
    )
    correct_response = chat_response.choices[0].message.content

    return correct_response


def get_item_categories():
    return [
        'Film & Animation',
        'Autos & Vehicles',
        'Music',
        'Pets & Animals',
        'Sports',
        'Travel & Events',
        'Gaming',
        'People & Blogs',
        'Comedy',
        'Entertainment',
        'News & Politics',
        'Howto & Style',
        'Education',
        'Science & Technology',
        'Nonprofits & Activism']

def L2norm(array):
    return array / np.linalg.norm(array, ord=2)


def save_new_item_click_records(round_cnt, name, new_item_click, config):
    with open(
            f'/home/xiaopeng_ye/experiment/Agent4Fairness/figures/prospect_theory/0929_DIN_{config["provider_decision_policy"]}_{config["reranking_model"]}_item_recency_{config["item_recency"]}_record.txt',
            'a') as f:
        f.write(f'{round_cnt}\t{name}\t{new_item_click}\n')  # 追加内容，不会换行
def save_action_records(round_cnt,name, new_item_click, exploit, config):
    with open(
            f'/home/xiaopeng_ye/experiment/Agent4Fairness/figures/prospect_theory/0929_DIN_{config["provider_decision_policy"]}_{config["reranking_model"]}_item_recency_{config["item_recency"]}_record.txt',
            'a') as f:
        if exploit == True:
            f.write(f'{round_cnt}\t{name}\t{new_item_click}\tEXPLOIT\n')  # 追加内容，不会换行
        else:
            f.write(f'{round_cnt}\t{name}\t{new_item_click}\tEXPLORE\n')  # 追加内容，不会换行