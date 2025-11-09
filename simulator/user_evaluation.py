import logging
import random

logging.basicConfig(level=logging.ERROR)
from datetime import datetime, timedelta, date
from typing import List
from termcolor import colored
import os
import logging
import argparse
from yacs.config import CfgNode
import csv
from tqdm import tqdm
import os
import time
import concurrent.futures
import json
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import faiss
import re
import dill
import numpy as np
import queue
from typing import List
import pandas as pd

from llamafactory.agents import RecAgent, ProAgent
from llamafactory.recommender.recommender import Recommender
from llamafactory.recommender.data.data import Data
from llamafactory.utils import utils
from llamafactory.utils import utils, message
from llamafactory.utils.message import Message
from llamafactory.utils.event import Event, update_event, reset_event
from llamafactory.utils import interval as interval
import threading
from llamafactory.agents.recagent_memory import RecAgentMemory, RecAgentRetriever
import heapq
from fastapi.middleware.cors import CORSMiddleware
from vllm import LLM as vvLLM
from vllm import SamplingParams
import transformers

import json


def read_json(file_path):
    """
    读取 JSON 文件并返回其内容。

    :param file_path: JSON 文件的路径
    :return: JSON 文件内容（以字典或列表的形式）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的 JSON 格式。")
    except Exception as e:
        print(f"发生错误：{e}")



class Simulator:
    """
    Simulator class for running the simulation.
    """

    def __init__(self, config: CfgNode, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.round_cnt = 0
        self.round_msg: List[Message] = []
        self.active_agents: List[int] = []  # active agents in current round
        self.active_proagents: List[int] = []
        self.active_agent_threshold = config["active_agent_threshold"]
        self.active_proagent_threshold = config["active_proagent_threshold"]
        self.active_method = config["active_method"]
        self.proagent_active_method = config["proagent_active_method"]
        self.file_name_path: List[str] = []
        self.play_event = threading.Event()
        self.working_agents: List[RecAgent] = []  # busy agents
        self.now = datetime.now().replace(hour=8, minute=0, second=0)
        self.interval = interval.parse_interval(config["interval"])
        self.round_entropy = []
        self.round_provider_num = []
        # self.rec_cnt = [20] * config["agent_num"]
        self.rec_stat = message.RecommenderStat(
            tot_user_num=0,
            cur_user_num=0,
            tot_item_num=0,
            inter_num=0,
            rec_model=config["rec_model"],
            pop_items=[],
        )


        self.new_round_item = []
        self.tokenizer, self.model = None, None
        self.embedding_size, self.embedding_model = utils.get_embedding_model()
        self.leave_providers = []

        # self.llm = vvLLM(model="/home/xiaopeng_ye/LLMs/Meta-Llama-3-8B-Instruct",
        #                  tensor_parallel_size=2,
        #                  # trust_remote_code=True,
        #                  )
        # self.tokenizer = AutoTokenizer.from_pretrained('/home/xiaopeng_ye/LLMs/Meta-Llama-3-8B-Instruct')
        if os.path.exists(self.config['profile_path']):
            with open(self.config['profile_path'], 'r') as f:
                self.provider_profile_dict = json.load(f)
        else:
            self.provider_profile_dict = {}



    def get_file_name_path(self):
        return self.file_name_path

    def load_simulator(self):
        """Load and initiate the simulator."""
        self.round_cnt = 0
        # self.embedding_model = utils.get_embedding_model()
        self.data = Data(self.config)
        self.agents = self.agent_creation()
        # self.provider_agents = self.provider_agent_creation()
        # self.recsys = Recommender(self.config, self.logger, self.data)
        self.logger.info("Simulator loaded.")
        self.logger.info(f'Config :{self.config}')




    def relevance_score_fn(self, score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # This will differ depending on a few things:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
        # This function converts the euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - score / math.sqrt(2)

    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        # Define your embedding model
        embedding_size, embeddings_model = self.embedding_size, self.embedding_model
        # Initialize the vectorstore as empty
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
        )

        # If choose RecAgentMemory, you must use RecAgentRetriever rather than TimeWeightedVectorStoreRetriever.
        RetrieverClass = (
            RecAgentRetriever
            # if self.config["recagent_memory"] == "recagent"
            # else TimeWeightedVectorStoreRetriever
        )

        return RetrieverClass(
            vectorstore=vectorstore, other_score_keys=["importance"], now=self.now, k=5
        )

    def pause(self):
        self.play_event.clear()

    def play(self):
        self.play_event.set()



    def one_step_for_user_with_rec(self, agent_id, item_ids, rec_items):
        """Run one step of an agent."""
        # self.play_event.wait()
        # if not self.check_active(agent_id):
        #     return [
        #         Message(agent_id=agent_id, action="NO_ACTION", content="No action.")
        #     ]
        agent = self.agents[agent_id]
        name = agent.name
        interest = agent.interest
        message = []
        # with lock:
        #     heapq.heappush(self.working_agents, agent)
        # if "REC" in choice:
        ids = []  # 被推荐的商品

        duration = 2
        click_list = []
        for true_rec_itemid, true_rec_item_description in zip(item_ids, rec_items): #
            observation = f"{name} is browsing the recommender system."

            observation = (
                observation
                + f" {name} is recommended {true_rec_item_description}."
            )
            # choice, action = agent.take_recommender_action(observation, self.now)
            choice, action = agent.take_click_action_with_rec(observation, self.now)
            # print(f'-----choice:{choice}------action:{action}')
            ids.extend([true_rec_itemid])

            if choice =='[WATCH]':
                agent.update_watched_history(true_rec_item_description)
                click_list.append(1)
            elif choice == '[SKIP]':
                # self.logger.info(f"{name} ({interest}) skip the video.")
                click_list.append(0)
            else:
                self.logger.info(f"{name} leaves the recommender system.")
                click_list.append(0)
                break
        if len(click_list) < len(item_ids):
            click_list.extend([0] * (len(item_ids) - len(click_list)))
        # print(click_list)
        return click_list


    def check_active(self, index: int):
        # If agent's previous action is completed, reset the event
        agent = self.agents[index]
        if (
            self.active_agent_threshold
            and len(self.active_agents) >= self.active_agent_threshold
        ):
            return False

        active_prob = agent.get_active_prob(self.active_method)
        if np.random.random() > active_prob:
            agent.no_action_round += 1 # 随机数大于 则 没action
            return False
        self.active_agents.append(index)
        return True

    def get_user_feedbacks(self, TopK=20, neg_num=10):
        user_list = [u for u in self.active_agents if len(self.data.users[u]['history']) >=20]
        print(f'active user num:{len(user_list)}')
        recommend_lists = []
        true_items = []
        for u in tqdm(user_list):
            recommend_list = []
            # print(self.data.users)
            true_item = random.sample(self.data.users[u]['history'], TopK-neg_num)
            recommend_list.extend(true_item)
            true_items.append(true_item)
            filtered_list = [item for item in list(self.data.items.keys()) if item != true_item]
            recommend_list.extend(random.sample(filtered_list, neg_num))
            random.shuffle(recommend_list)
            recommend_lists.append(recommend_list)
        users_sorted_items = {}
        users_item_decs = {}

        for user, user_recommend_list in zip(user_list, recommend_lists):
            sorted_items = [item for item in user_recommend_list]
            # print(sorted_items)
            sorted_item_names = self.data.get_item_names(sorted_items)
            sorted_item_tags = self.data.get_item_tags(sorted_items)
            description = self.data.get_item_description_by_id(sorted_items)
            items = [
                sorted_item_names[i]
                + ";; Genre: "
                + self.data.get_genres_by_ids([sorted_items[i]])[0]
                + ";; Tags: "
                + sorted_item_tags[i]
                + ";; Description: "
                + description[i]
                for i in range(len(sorted_item_names))
            ]
            users_sorted_items[user] = sorted_items  # for user index in users[index]
            users_item_decs[user] = items

        click_results = []
        if self.config["execution_mode"] == "parallel":
            batch_size = 200
            futures = []
            with tqdm(total=len(user_list), desc='User Processing...') as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                    for i in user_list:
                        item_ids = users_sorted_items[i]  #[:self.config['TopK']]
                        rec_items = users_item_decs[i]  #[:self.config['TopK']]
                        futures.append(executor.submit(self.one_step_for_user_with_rec,
                                                       i,
                                                       item_ids,
                                                       rec_items))

                    for future in concurrent.futures.as_completed(futures):
                        click_result = future.result()
                        click_results.append(click_result)
                        # print(click_results)
                        pbar.update(1)

        return [users_sorted_items[u] for u in user_list], click_results, true_items


    def create_agent(self, i, api_key) -> RecAgent:
        """
        Create an agent with the given id.
        """
        LLM = utils.get_llm(config=self.config,
                            logger=self.logger,
                            api_key=api_key)
        MemoryClass = (
            RecAgentMemory
            # if self.config["recagent_memory"] == "recagent"
            # else GenerativeAgentMemory
        )

        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            now=self.now,
            verbose=False,
            reflection_threshold=10,
            embedding_model=self.embedding_model
        )
        agent = RecAgent(
            id=i,
            name=self.data.users[i]["name"],
            # age=self.data.users[i]["age"],
            # gender=self.data.users[i]["gender"],
            # traits=self.data.users[i]["traits"],
            # status=self.data.users[i]["status"],
            interest=self.data.users[i]["interest"],
            # relationships=self.data.get_relationship_names(i),
            # feature=utils.get_feature_description(self.data.users[i]["feature"]),
            memory_retriever=None, #self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=reset_event(self.now),
        )
        # observations = self.data.users[i]["observations"].strip(".").split(".")
        # for observation in observations:
        #     agent.memory.add_memory(observation, now=self.now)
        return agent


    def agent_creation(self):
        """
        Create agents in parallel
        """
        agents = {}
        api_keys = list(self.config["api_keys"])
        agent_num = int(self.config["agent_num"])
        # Add ONE user controllable user into the simulator if the flag is true.
        # We block the main thread when the user is creating the role.
        for i in tqdm(range(1, agent_num+1)):
            api_key = api_keys[i % len(api_keys)]
            agent = self.create_agent(i, api_key)
            agent.active_prob = 1
            agents[agent.id] = agent

        return agents

    def update_round(self):
        self.active_proagents.clear()
        self.active_agents.clear()
        # self.recsys.round_record.clear()

        for i in tqdm(range(1, self.config["agent_num"] + 1)):
            if self.check_active(i):
                continue
        self.logger.info(f'active provider number:{len(self.active_proagents)}')
        self.logger.info(f'active user number:{len(self.active_agents)}')


def evaluate(users_sorted_items, click_results, true_items):
    acc_list = []
    pre_list = []
    recall_list = []
    f1_list = []
    for item_ids, click_result, true_item in zip(users_sorted_items, click_results, true_items):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for i, y in zip(item_ids, click_result):
            if i in true_item and y == 1:
                true_positive += 1
            elif i not in true_item and y == 0:
                true_negative += 1
            elif i in true_item and y == 0:
                false_negative += 1
            elif i not in true_item and y == 1:
                false_positive += 1
        print(true_positive, true_negative, false_positive, false_negative)
        precision = true_positive/(true_positive + false_positive) if (true_positive + false_positive) != 0 else None
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else None
        acc = (true_positive + true_negative) / len(item_ids)
        if precision is None or recall is None:
            f1 = None
        else:
            f1 = 2*(precision * recall) / (precision + recall) if (precision + recall) != 0 else None
        if acc is None or precision is None or recall is None or f1 is None:
            continue
        if np.isnan(acc) or np.isnan(precision) or np.isnan(recall) or np.isnan(f1):
            continue
        acc_list.append(acc)
        pre_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    avg_accuracy = np.mean(acc_list)
    avg_pre = np.mean(pre_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    return avg_accuracy, avg_pre, avg_recall, avg_f1




if __name__ == '__main__':
    import wandb
    # intialize_the_simulator
    config = CfgNode(new_allowed=True)  # config/config.yaml
    config.merge_from_file('/home/xiaopeng_ye/experiment/Agent4Fairness/LLaMA-Factory/src/llamafactory/config/config.yaml')
    #
    logger = utils.set_logger('simulation.log', '0630')
    logger.info(f"simulator config: \n{config}")
    # logger.info(f"os.getpid()={os.getpid()}")
    simulator = Simulator(config, logger)
    simulator.load_simulator()
    simulator.update_round()
    acc_list = []
    pre_list = []
    recall_list = []
    f1_list = []
    for i in range(5):
        users_sorted_items, click_results, true_items = simulator.get_user_feedbacks(TopK=20,
                                                                                     neg_num=18)
        # print(users_sorted_items, click_results)
        avg_accuracy, avg_pre, avg_recall, avg_f1 = evaluate(users_sorted_items, click_results, true_items)
        acc_list.append(avg_accuracy)
        pre_list.append(avg_pre)
        recall_list.append(avg_recall)
        f1_list.append(avg_f1)

    print(f'Accuracy:{acc_list} \n \t Mean:{np.mean(acc_list)} Std:{np.std(acc_list)}')
    print(f'Precision:{pre_list} \n \t Mean:{np.mean(pre_list)} Std:{np.std(pre_list)}')
    print(f'Recall:{recall_list} \n \t Mean:{np.mean(recall_list)} Std:{np.std(recall_list)}')
    print(f'F1:{f1_list} \n \t Mean:{np.mean(f1_list)} Std:{np.std(f1_list)}')