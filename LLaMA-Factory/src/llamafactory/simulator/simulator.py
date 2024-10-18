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
lock = threading.Lock()
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_seed(seed):
    '''setting random seeds'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(2024)



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
        self.provider_agents = self.provider_agent_creation()
        self.recsys = Recommender(self.config, self.logger, self.data)
        self.logger.info("Simulator loaded.")
        self.logger.info(f'Config :{self.config}')


        for uid, u_dict in self.data.users.items():
            for iid in u_dict['history']:
                self.recsys.add_train_data(uid, iid, 1)
            all_item_id = list(self.data.items.keys())
            rest_item_id = [i for i in all_item_id if i not in u_dict['history']]
            for iid in random.sample(rest_item_id, 10):
                self.recsys.add_train_data(uid, iid, 0)



    def save(self, save_dir_name):
        """Save the simulator status of current round"""
        utils.ensure_dir(save_dir_name)
        ID = utils.generate_id(self.config["simulator_dir"])
        file_name = f"{ID}-Round[{self.round_cnt}]-AgentNum[{self.config['agent_num']}]-{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}"
        self.file_name_path.append(file_name)
        save_file_name = os.path.join(save_dir_name, file_name + ".pkl")
        with open(save_file_name, "wb") as f:
            dill.dump(self.__dict__, f)
        self.logger.info("Current simulator Save in: \n" + str(save_file_name) + "\n")
        self.logger.info(
            "Simulator File Path (root -> node): \n" + str(self.file_name_path) + "\n"
        )
        utils.ensure_dir(self.config["ckpt_path"])
        cpkt_path = os.path.join(self.config["ckpt_path"], file_name + ".pth")
        self.recsys.save_model(cpkt_path)
        self.logger.info(
            "Current Recommender Model Save in: \n" + str(cpkt_path) + "\n"
        )

    @classmethod
    def restore(cls, restore_file_name, config, logger):
        """Restore the simulator status from the specific file"""
        with open(restore_file_name + ".pkl", "rb") as f:
            obj = cls.__new__(cls)
            obj.__dict__ = dill.load(f)
            obj.config, obj.logger = config, logger
            return obj

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


    def global_message(self, message: str):
        for i, agent in self.agents.items():
            agent.memory.add_memory(message, self.now)

    def one_step_for_provider(self, agent_id):
        """Run one step of an agent."""
        # self.play_event.wait()
        proagent = self.provider_agents[agent_id]
        # proagent.initialize_tok_mod(self.tokenizer, self.model)
        name = proagent.name
        # if self.round_cnt == 1:
        #     proagent.initialize_provider_profile()
        proagent.update_round(self.round_cnt)
        # if not self.check_proactive(agent_id):
        #     return message
        user_interest_dict = self.data.get_user_interest_dict()
        categories = self.data.get_item_categories()
        decision_policy = self.config['provider_decision_policy']
        # if len(proagent.new_round_item) > 0:
        #     new_item_click =  proagent.item_acc_click[proagent.new_round_item[-1] ]
        #     utils.save_new_item_click_records(self.round_cnt, proagent.name, new_item_click, self.config)
        if decision_policy not in ['random', 'LBR', 'CFD']:
            analyze_prompt, analyze_result = proagent.analyzing(categories=categories,
                                                                user_interest_dict=user_interest_dict,
                                                                round_cnt=self.round_cnt)
        # self.logger.info(f'Provider {name} has analyzed for the user feedback, and have a conclusion {analyze_conclusion}')
            analyze_history = [analyze_prompt, analyze_result]  #None
            choose_genre, choice, action = proagent.extract_analyze_result(categories, analyze_result)


        else:
            analyze_result = ''
            analyze_prompt = ''
            analyze_history = None
            if decision_policy == 'random':
                choose_genre = random.choice(categories)
                choice = '[RAMDOM]'
                action = f'{proagent.name} choose to create video of genre {choose_genre}'
            elif decision_policy == 'LBR':
                choice = '[LBR]'
                item_num = len(proagent.new_round_item)
                creation_state = proagent.get_creation_state()
                if item_num != 0:
                    direction = np.random.randn(len(categories))  # 正太分布
                    direction = utils.L2norm(direction)
                    middle_state = creation_state + self.config['provider_lr'] * direction
                    user_vector = proagent.get_creation_utility_vector()
                    print(f'user vector:{user_vector}')
                    middle_utility = np.dot(middle_state, user_vector)
                    ori_utility = np.dot(creation_state, user_vector)
                    print(f'middle_utility:{middle_utility}')
                    print(f'ori utility:{ori_utility}')
                    if middle_utility > ori_utility:
                        proagent.creation_state = utils.L2norm(middle_state)
                print(f'{name} creation state:{proagent.creation_state}')
                max_index = np.argmax(proagent.creation_state)

                choose_genre = categories[max_index]
                action = f'{proagent.name} choose to create video of genre {choose_genre}'
            elif decision_policy == 'CFD':
                choice = '[CFD]'
                item_num = len(proagent.new_round_item)
                creation_state = proagent.get_creation_state()
                if item_num != 0:
                    user_vector = proagent.get_creation_utility_vector()
                    print(f'user vector:{user_vector}')
                    proagent.creation_state = creation_state + self.config['provider_lr'] * user_vector
                    proagent.creation_state = utils.L2norm(proagent.creation_state)
                print(f'{name} creation state:{proagent.creation_state}')
                max_index = np.argmax(proagent.creation_state)
                choose_genre = categories[max_index]
                action = f'{proagent.name} choose to create video of genre {choose_genre}'

            else:
                choose_genre = 'WRONG'
                choice = 'Wrong'
                raise ValueError(f'Not Valid decision_policy:{decision_policy}')
        if len(proagent.new_round_item) > 0:
            new_item_id = proagent.new_round_item[-1]
            new_item_click = proagent.item_acc_click[new_item_id]
            new_item_genre = proagent.items[new_item_id]['genre']
            if choose_genre == new_item_genre:
                exploit = True
            else:
                exploit = False
            utils.save_action_records(self.round_cnt, proagent.name, new_item_click, exploit, self.config)

        retries = 0
        while retries < 5:
            try:
                generate_result = proagent.generating(action, choice, choose_genre, analyze_history)
                item_name, item_genre, item_tags, item_description = utils.response_to_item(generate_result, choose_genre)
                self.upload_item(item_name, item_genre, item_tags, item_description, name)
                break
            except AttributeError as e:
                print(f"Error occurred: {e}, retrying...")
                if retries == 4:
                    raise ValueError('No valid item')
                retries += 1

        return message

    def upload_item(self, name, genre, tags,  description, provider_name):
        provider_id = self.data.get_provider_id_by_name(provider_name)
        # if name == '':
        #     self.new_round_item.append((provider_id, -1))
        #     return -1
        try:
            max_item_id = max(self.data.get_all_item_ids())
        except:
            max_item_id = 0
        # print(f'all item id:{self.data.get_all_item_ids()}')
        # print(f'max_item_id:{max_item_id}')
        new_item_id = int(max_item_id) + 1   # 1 is the first item
        self.data.items[new_item_id] = {
            "name": name.strip(),
            "provider_name": provider_name,
            "provider_id": provider_id,
            "genre": genre,
            "upload_time": self.round_cnt,
            "tags": tags,
            "description": description.strip(),
            "inter_cnt": 0,
            "mention_cnt": 0,
        }
        self.data.providers[provider_id]['items'].append(new_item_id)
        self.recsys.data = self.data
        self.data.item2provider[new_item_id] = provider_id
        self.provider_agents[provider_id].add_item(new_item_id, self.data.items[new_item_id])
        self.new_round_item.append((provider_id, new_item_id))
        self.logger.info(f'<{provider_name}> create a new item:\n NAME:{name.strip()} GENRE:{genre} DESC:{description.strip()}')
        return new_item_id

    # def clear_items(self):
    #     self.data.items = {}

    def get_recent_items(self, item_recency):
        items = self.data.items
        recent_item_ids = []
        for item_id, item_dict in items.items():
            up_time = item_dict['upload_time']
            if self.round_cnt - up_time <= item_recency:
                recent_item_ids.append(item_id)
        return recent_item_ids

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
        # self.rec_cnt[agent_id] += 1
        # self.logger.info(f"{name} enters the recommender system.")
        # self.round_msg.append(
        #     Message(
        #         agent_id=agent_id,
        #         action="RECOMMENDER",
        #         content=f"{name} enters the recommender system.",
        #     )
        # )
        # if self.round_cnt == 1:
        #     new_items = [i for p, i in self.new_round_item]
        #     # print(f'new items:{new_items}')
        #     item_ids, rec_items = self.recsys.get_random_items(agent_id, items=new_items)
        # else:
        # new_items = [i for p, i in self.new_round_item]
        # recent_items = self.get_recent_items()
        # item_ids, rec_items = self.recsys.get_full_sort_items(agent_id, item_set=recent_items)      # 要推荐的item。按顺序排列  itemids 是id， recitems 是item的描述
        # item_ids = item_ids[:self.config['TopK']]
        # rec_items = rec_items[:self.config['TopK']]
        # self.logger.info(f'rec items:{rec_items}')

        duration = 2
        for true_rec_itemid, true_rec_item_description in zip(item_ids, rec_items): #
            # self.logger.info(
            #     f"{name} is recommended {true_rec_item_description}."
            # )
            # self.round_msg.append(
            #     Message(
            #         agent_id=agent_id,
            #         action="RECOMMENDER",
            #         content=f"{name} is recommended {true_rec_item_description}.",
            #     )
            # )
            observation = f"{name} is browsing the recommender system."

            observation = (
                observation
                + f" {name} is recommended {true_rec_item_description}."
            )
            # choice, action = agent.take_recommender_action(observation, self.now)
            choice, action = agent.take_click_action_with_rec(observation, self.now)
            # print(f'-----choice:{choice}------action:{action}')
            self.recsys.update_history_by_id(
                agent_id,
                [true_rec_itemid],
            )
            ids.extend([true_rec_itemid])

            if choice =='[WATCH]':
                # self.logger.info(f"{name} ({interest}) watches {true_rec_item_description}")
                # message.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} watches {true_rec_item_description}.",
                #     )
                # )
                # self.round_msg.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} watches {true_rec_item_description}.",
                #     )
                # )
                agent.update_watched_history(true_rec_item_description)
                self.recsys.update_positive_by_id(agent_id, true_rec_itemid)
                self.recsys.add_round_record(agent_id, true_rec_itemid, 1, self.round_cnt)
                self.recsys.add_train_data(agent_id, true_rec_itemid, 1)

                self.update_click_to_providers(true_rec_itemid)  # 点击更新给供应商
                self.update_exposure_to_providers(true_rec_itemid)


                # item_descriptions = self.data.get_item_description_by_name([true_rec_item_description])

                # observation = f"{name} has just finished watching {true_rec_item_description};;{item_descriptions[0]}."
                # feelings = agent.generate_feeling(
                #     observation, self.now + timedelta(hours=duration)
                # )
                # provider_id = self.data.get_provider_id_by_item_id(true_rec_itemid)
                # self.provider_agents[provider_id].upload_comments(feelings)
                # self.logger.info(f"{name}({interest}) feels: {feelings}")
                #
                # self.round_msg.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} feels: {feelings}",
                #     )
                # )
            elif choice == '[SKIP]':
                # self.logger.info(f"{name} ({interest}) skip the video.")
                self.recsys.add_train_data(
                    agent_id, true_rec_itemid, 0
                )
                self.recsys.add_round_record(agent_id, true_rec_itemid, 0, self.round_cnt)
                self.update_exposure_to_providers(true_rec_itemid)
                # self.round_msg.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} looks next page.",
                #     )
                # )
            else:
                self.logger.info(f"{name} leaves the recommender system.")
                # self.round_msg.append(
                #     Message(
                #         agent_id=agent_id,
                #         action="RECOMMENDER",
                #         content=f"{name} leaves the recommender system.",
                #     )
                # )
                break
        # self.recsys.round_record[agent_id].append(ids)

        return message

    def one_step_for_user(self, agent_id):
        """Run one step of an agent."""
        self.play_event.wait()
        # if not self.check_active(agent_id):
        #     return [
        #         Message(agent_id=agent_id, action="NO_ACTION", content="No action.")
        #     ]
        agent = self.agents[agent_id]
        name = agent.name
        interest = agent.interest
        message = []
        # choice, observation = agent.take_action(self.now)  # 选择进入RS，进入社交媒体，还是什么都不做
        # with lock:
        #     heapq.heappush(self.working_agents, agent)
        # if "REC" in choice:
        ids = []  # 被推荐的商品
        # self.rec_cnt[agent_id] += 1
        self.logger.info(f"{name} enters the recommender system.")
        self.round_msg.append(
            Message(
                agent_id=agent_id,
                action="RECOMMENDER",
                content=f"{name} enters the recommender system.",
            )
        )
        # item_ids, rec_items = self.recsys.get_full_sort_items(agent_id)    # 要推荐的item。按顺序排列  itemids 是id， recitems 是item的描述
        new_items = [i for p, i in self.new_round_item]
        # print(f'new items:{new_items}')
        item_ids, rec_items = self.recsys.get_random_items(agent_id, items=new_items)
        duration = 2
        for true_rec_itemid, true_rec_item_description in zip(item_ids, rec_items):
        # while not leave:
            self.logger.info(
                f"{name}({interest}) is recommended {true_rec_item_description}."
            )
            self.round_msg.append(
                Message(
                    agent_id=agent_id,
                    action="RECOMMENDER",
                    content=f"{name} is recommended {true_rec_item_description}.",
                )
            )
            observation = f"{name} is browsing the recommender system."

            observation = (
                observation
                + f" {name} is recommended {true_rec_item_description}."
            )
            # choice, action = agent.take_recommender_action(observation, self.now)
            choice, action = agent.take_click_action(observation, self.now)
            # print(f'-----choice:{choice}------action:{action}')
            self.recsys.update_history_by_id(
                agent_id,
                [true_rec_itemid],
            )
            ids.extend([true_rec_itemid])

            if choice =='[WATCH]':

                self.logger.info(f"{name}({interest}) watches {true_rec_item_description}")
                message.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} watches {true_rec_item_description}.",
                    )
                )
                self.round_msg.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} watches {true_rec_item_description}.",
                    )
                )
                agent.update_watched_history(true_rec_item_description)
                self.recsys.update_positive_by_id(agent_id, true_rec_itemid)

                # for i in range(self.recsys.page_size):
                #     # print(f'action:{action}')
                #     # print(f'666{item_ids[page * self.recsys.page_size + i]}')
                #     try:
                #         exposed_item_id = item_ids[page * self.recsys.page_size + i]
                #     except IndexError:
                #         continue
                # if i == action - 1:
                # update
                # print(f'yes : i inter')
                self.recsys.add_train_data(agent_id, true_rec_itemid, 1)
                # self.recsys.add_round_provider_click_data()
                # print(f'update click to provider:{true_rec_itemid}')
                self.update_click_to_providers(true_rec_itemid)  # 点击更新给供应商
                self.update_exposure_to_providers(true_rec_itemid)
                # else:
                #     self.recsys.add_train_data(agent_id, exposed_item_id, 0)
                #     self.update_exposure_to_providers(exposed_item_id) # 把曝光更新给每一个供应商


                item_descriptions = self.data.get_item_description_by_name([true_rec_item_description])

                observation = f"{name} has just finished watching {true_rec_item_description};;{item_descriptions[0]}."
                feelings = agent.generate_feeling(
                    observation, self.now + timedelta(hours=duration)
                )
                provider_id = self.data.get_provider_id_by_item_id(true_rec_itemid)
                self.provider_agents[provider_id].upload_comments(feelings)
                self.logger.info(f"{name}({interest}) feels: {feelings}")

                self.round_msg.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} feels: {feelings}",
                    )
                )

            else:
                self.logger.info(f"{name}({interest}) skip the video.")
                self.recsys.add_train_data(
                    agent_id, true_rec_itemid, 0
                )
                self.update_exposure_to_providers(true_rec_itemid)
                self.round_msg.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} looks next page.",
                    )
                )

        self.logger.info(f"{name} leaves the recommender system.")
        self.round_msg.append(
            Message(
                agent_id=agent_id,
                action="RECOMMENDER",
                content=f"{name} leaves the recommender system.",
            )
        )

        # self.recsys.round_record[agent_id].append(ids)

        return message


    def update_exposure_to_providers(self, exposed_item_id):
        # print(f'item2provider:{self.item2provider}')
        belong_provider_agent = self.provider_agents[self.data.item2provider[exposed_item_id]]
        # print(f'update exposure to {belong_provider_agent.name}')
        belong_provider_agent.update_exposure(exposed_item_id, self.round_cnt)

    def update_click_to_providers(self, clicked_item_id):

        belong_provider_agent = self.provider_agents[self.data.item2provider[clicked_item_id]]
        # print(f'update click to {belong_provider_agent.name}')
        belong_provider_agent.update_click(clicked_item_id, self.round_cnt)


    def construct_generate_prompts(self, analyze_responses_list, analyze_prompt_list):
        data_list = []
        choose_genre = ''
        for i, pro_id in tqdm(enumerate(self.active_proagents)): #tqdm(range(1, self.config["provider_agent_num"]+1)):
            proagent = self.provider_agents[pro_id]
            name = proagent.name
            profile_text = proagent.get_profile_text()

            categories = utils.get_item_categories()
            decision_policy = self.config['provider_decision_policy']
            if decision_policy not in ['random', 'LBR', 'CFD']:
                analyze_result = analyze_responses_list[i]
                analyze_prompt = analyze_prompt_list[i]
                choose_genre, choice, action = proagent.extract_analyze_result(categories, analyze_result)
            else:
                analyze_result = ''
                analyze_prompt = ''
                if decision_policy == 'random':
                    choose_genre = random.choice(categories)
                    choice = '[RAMDOM]'
                    action = f'{proagent.name} choose to create video of genre {choose_genre}'
                elif decision_policy == 'LBR':
                    choice = '[LBR]'
                    item_num = len(proagent.new_round_item)
                    creation_state = proagent.get_creation_state()
                    if item_num != 0:
                        direction = np.random.randn(len(categories))  # 正太分布
                        direction = utils.L2norm(direction)
                        middle_state = creation_state + self.config['provider_lr'] * direction
                        user_vector = proagent.get_creation_utility_vector()
                        print(f'user vector:{user_vector}')
                        middle_utility = np.dot(middle_state, user_vector)
                        ori_utility = np.dot(creation_state, user_vector)
                        print(f'middle_utility:{middle_utility}')
                        print(f'ori utility:{ori_utility}')
                        if middle_utility > ori_utility:
                            proagent.creation_state = utils.L2norm(middle_state)
                    print(f'{name} creation state:{proagent.creation_state}')
                    max_index = np.argmax(proagent.creation_state)

                    choose_genre = categories[max_index]
                    action = f'{proagent.name} choose to create video of genre {choose_genre}'
                elif decision_policy == 'CFD':
                    choice = '[CFD]'
                    item_num = len(proagent.new_round_item)
                    creation_state = proagent.get_creation_state()
                    if item_num != 0:
                        user_vector = proagent.get_creation_utility_vector()
                        print(f'user vector:{user_vector}')
                        proagent.creation_state = creation_state + self.config['provider_lr'] * user_vector
                        proagent.creation_state = utils.L2norm(proagent.creation_state)
                    print(f'{name} creation state:{proagent.creation_state}')
                    max_index = np.argmax(proagent.creation_state)
                    choose_genre = categories[max_index]
                    action = f'{proagent.name} choose to create video of genre {choose_genre}'


                else:
                    choose_genre = 'WRONG'
                    choice =  'Wrong'
                    raise ValueError(f'Not Valid decision_policy:{decision_policy}')

            # print(choice)
            # print(choose_genre)
            # print(action)
            recent_creation = proagent.get_recent_creation(category=choose_genre)
            single_dict = dict({"system": profile_text,
                                "instruction":  (
                                                f"Based on the analysis, {action}"
                                                f"\n Please create a brand new item in the {choose_genre} genre."
                                                "\n Return the results strictly according to the following JSON dictionary format: \n ```json"
                                                '''\n {"name": "item_name", "genre": "''' + choose_genre + '''", "tags": [tag1, tag2, tag3], "description": "item_description_text"}'''
                                ),
                                "input": f"You can draw inspiration from {name}'s previous creation on genre {choose_genre}, but cannot replicate them identically.\n{recent_creation}" if (recent_creation != None and choice != '[RANDOM]') else "",
                                         # f"{name} recently created {proagent.items[proagent.new_round_item[-1]]} on recommender system.",
                                "output": "",
                                "history": [
                                    [analyze_prompt, analyze_result],
                                ]}
                               )
            data_list.append(single_dict)

        # with open(f'round{self.round_cnt}.json', "w") as file:
        #     json.dump(data_list, file)

        return data_list, choose_genre

    def provider_round(self):
        if self.config["execution_mode"] == "parallel":
            batch_size = 200
            futures = []
            with tqdm(total=len(self.active_proagents), desc='Provider Processing...') as pbar:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for i in self.active_proagents:
                        futures.append(executor.submit(self.one_step_for_provider, i))
                    # for b in range(1, self.config["provider_agent_num"] + 1, batch_size):
                    #     futures = [executor.submit(self.one_step_for_provider, i) for i in
                    #                range(b, min(self.config["provider_agent_num"] + 1, b + batch_size))]
                        # 等待当前批次完成
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        pbar.update(1)
                            # print(f'mission complete:{result}')
        else:
            for i in tqdm(self.active_proagents):
                self.one_step_for_provider(i)

        # self.logger.info(f'active provider number:{len(self.active_proagents)}')

    def update_round(self, round_cnt):
        self.round_cnt = round_cnt + 1
        self.new_round_item.clear()
        self.active_proagents.clear()
        self.active_agents.clear()
        self.recsys.round_record.clear()

        if self.config['rec_model'] not in ['Random']:
            if self.round_cnt % 5 == 1:
                if self.config['rec_model'] == 'BPR':
                    self.recsys.train_BPR()
                else:
                    self.recsys.train()


        for i in tqdm(range(1, self.config["provider_agent_num"]+1)):
            proagent = self.provider_agents[i]
            name = proagent.name
            proagent.update_round(self.round_cnt)
                # self.active_proagents.append(i)
            if self.round_cnt == 1:
                if name in self.provider_profile_dict.keys():
                    profile_text = proagent.initialize_provider_profile(profile=self.provider_profile_dict[name])
                else:
                    profile_text = proagent.initialize_provider_profile(profile=None)
                    self.provider_profile_dict[name] = profile_text
                    with open(self.config['profile_path'], 'w') as f:
                        json.dump(self.provider_profile_dict, f, indent=4)
            if self.check_proactive(i):
                continue
        for i in tqdm(range(1, self.config["agent_num"] + 1)):
            if self.check_active(i):
                continue
        self.logger.info(f'active provider number:{len(self.active_proagents)}')
        self.logger.info(f'active user number:{len(self.active_agents)}')

    def construct_analyze_prompts(self):
        data_list = []
        for i in tqdm(self.active_proagents): #tqdm(range(1, self.config["provider_agent_num"]+1)):
            proagent = self.provider_agents[i]
            name = proagent.name
            profile_text = proagent.get_profile_text()

            categories = self.data.get_item_categories()
            prompt = proagent.get_analyze_prompt(categories=categories,
                                                 round_cnt=self.round_cnt,
                                                 user_interest_dict=self.data.get_user_interest_dict())
            single_dict = dict({"system": profile_text,
                               "instruction": prompt,
                               "input": '',
                               "output": "",
                                "history": []
                                })
            data_list.append(single_dict)

        # with open(f'round{self.round_cnt}.json', "w") as file:
        #     json.dump(data_list, file)

        return data_list

    def get_genre_item_count(self):
        cates = self.data.get_item_categories()
        count_dict = {k: 0 for k in cates}
        for pro_id, item_id in self.new_round_item:
            genre = self.data.items[item_id]['genre']
            count_dict[genre] += 1
        return count_dict


    def check_proactive(self, index: int):
        # If agent's previous action is completed, reset the event
        proagent = self.provider_agents[index]
        if (
            self.active_proagent_threshold
            and len(self.active_proagents) >= self.active_proagent_threshold
        ):
            return False

        if self.config['with_leave'] and index in self.leave_providers:
            return False

        active_prob = proagent.get_active_prob(self.active_method)
        random_state = np.random.random()
        # print(f'random state:{random_state} \t active prob :{active_prob}')
        if random_state > active_prob:
            proagent.no_action_round += 1  # 随机数大于 则 没action
            return False  # 不 active
        # 如果过去几轮创作的item，item一个点击都没有，则离开
        if self.round_cnt >= self.config['reranking_start_step'] + 5:
            if self.config['with_leave'] and len(proagent.new_round_item) >= 5:
                acc_click = 0
                for item_id in proagent.new_round_item[-5:]:
                    acc_click += proagent.item_acc_click[item_id]
                if acc_click == 0:
                    self.logger.info(f'Round {self.round_cnt}: provider <{proagent.name}> Leaves Forever...')
                    proagent.active_prob = 0.0
                    self.leave_providers.append(index)
                    self.logger.info(f'Round {self.round_cnt}: leave providers {self.leave_providers}')
                    return False
        self.active_proagents.append(index)
        return True
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

    def get_user_feedbacks(self):
        recent_items = self.get_recent_items(item_recency=self.config['item_recency'])
        if self.config['rec_model'] == "DIN":
            item_ids_dict, rec_items_dict = self.recsys.get_full_sort_items_for_users(user_list=self.active_agents,
                                                                                  round_cnt=self.round_cnt,
                                                                                  item_set=recent_items)               # 要推荐的item。按顺序排列  itemids 是id， recitems 是item的描述
        elif self.config['rec_model'] in  ["MF", 'Random', 'BPR', 'Pop']:
            item_ids_dict, rec_items_dict = self.recsys.get_full_sort_items_for_MF(user_list=self.active_agents,
                                                                                   round_cnt=self.round_cnt,
                                                                                  item_set=recent_items)


        if self.config["execution_mode"] == "parallel":
            batch_size = 200
            futures = []
            with tqdm(total=len(self.active_agents), desc='User Processing...') as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                    for i in self.active_agents:
                        item_ids = item_ids_dict[i]  #[:self.config['TopK']]
                        rec_items = rec_items_dict[i]  #[:self.config['TopK']]
                        futures.append(executor.submit(self.one_step_for_user_with_rec,
                                                       i,
                                                       item_ids,
                                                       rec_items))

                    for future in concurrent.futures.as_completed(futures):
                        msgs = future.result()
                        pbar.update(1)
                # for b in range(1, self.config["agent_num"]+1, batch_size):
                #     futures = [executor.submit(self.one_step_for_user_with_rec, i) for i in range(b, min(self.config["agent_num"]+1, b+batch_size))]
                #     # 等待当前批次完成
                #     for future in concurrent.futures.as_completed(futures):
                #         result = future.result()
                #         # print(f'mission complete:{result}')
        else:
            for i in tqdm(self.active_agents, desc='User Processing...'):
                item_ids = item_ids_dict[i]  #[:self.config['TopK']]
                rec_items = rec_items_dict[i]  #[:self.config['TopK']]
                self.one_step_for_user_with_rec(i, item_ids, rec_items)

        # self.logger.info(f'active user number:{len(self.active_agents)}')
        self.recsys.save_round_interaction(self.round_cnt)
        # self.recsys.train()
        # self.save(os.path.join(self.config["simulator_dir"]))


        feedbacks = {}
        ctr_feedbacks = {}
        total_click = 0
        rewards = []

        for provider_id in self.active_proagents:
            proagent = self.provider_agents[provider_id]
            name = proagent.name
            if len(proagent.new_round_item) <=1:
                continue
            last_new_item = proagent.new_round_item[-2]
            last_new_item_click = proagent.item_acc_click[last_new_item]

            upload_time = self.data.items[last_new_item]['upload_time']
            item_age = self.round_cnt - upload_time
            if item_age >= self.config['item_recency']:
                active_round =  self.config['item_recency']
            else:
                active_round = item_age
            item_reward_per_round = last_new_item_click/active_round
            feedbacks[f'{name}-click'] = item_reward_per_round
            rewards.append(item_reward_per_round)

        if len(rewards) == 0:
            return 0
        else:
            return np.mean(rewards)

        # for provider_id, item_id in self.new_round_item:
        #     # if item_id == -1:
        #     #     feedbacks.append(0)
        #     # else:
        #     # print(f'item id:{id}')
        #     round_click = self.provider_agents[provider_id].item_acc_click[item_id]
        #     round_expo = self.provider_agents[provider_id].item_acc_click[item_id]
        #
        #     total_click += round_click
        #
        #     name = self.provider_agents[provider_id].name
        #     feedbacks[f'{name}-click'] = round_click
        #     rewards.append((round_click - 15)*0.5)
        #     if round_expo != 0:
        #         ctr_feedbacks[f'{name}-ctr'] = round_click/round_expo
        #
        #     # if round_expo == 0:
        #     #     feedbacks.append(0)
        #     # else:
        #     #     pos = round_click
        #     #     naga = round_expo - round_click
        #     #     feedbacks.append(pos - naga)
        #         # round_click /round_expo
        # # for i in tqdm(range(self.config["provider_agent_num"])):
        # #     print(f'provider round click: {self.provider_agents[i].item_round_click}')
        #
        #     # total_round_exposure = sum(list(self.provider_agents[i].item_round_exposure[-1].values()))
        #     # feedbacks[i] = total_round_click#/total_round_exposure  # average CTR for each provider
        # # feedbacks = [(f + 4) * 1 for f in feedbacks]#utils.standardize(feedbacks)
        # ori_click_dict = feedbacks
        # # reward = [(f - 15)*0.5 for f in feedbacks.values()]
        # # print(f'-------round:{self.round_cnt}  feedbacks:{feedbacks}')
        # # print(f'total_click:{total_click}')
        # self.logger.info(f'provider clicks:{ori_click_dict}')
        # self.logger.info(f'provider ctrs:{ctr_feedbacks}')
        # return ori_click_dict, rewards, total_click, ctr_feedbacks



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

    def create_provider_agent(self, i, api_key) -> ProAgent:
        """
        Create an agent with the given id.
        """
        LLM = utils.get_llm(config=self.config,
                            logger=self.logger,
                            api_key=api_key
                            )
        MemoryClass = (
            RecAgentMemory
            # if self.config["recagent_memory"] == "recagent"
            # else GenerativeAgentMemory
        )

        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(), #utils.get_embedding_model(),
            now=self.now,
            verbose=False,
            reflection_threshold=10,
            embedding_model=self.embedding_model
        )
        agent = ProAgent(
            id=i,
            name=self.data.providers[i]["name"],
            # status=self.data.providers[i]["mood"],
            category_history=self.data.providers[i]["category_history"],
            items={id: self.data.items[id] for id in self.data.providers[i]['items']},
            memory_retriever=None, #self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=reset_event(self.now),
            config=self.config,
            active_prob=self.data.providers[i]["frequency"]

        )
        # print(self.data.items)
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

        if self.active_method == "random":
            active_probs = [self.config["active_prob"]] * agent_num
        else:
            active_probs = np.random.pareto(self.config["active_prob"] * 10, agent_num)
            active_probs = active_probs / active_probs.max()

        for i in tqdm(range(1, agent_num+1)):
            api_key = api_keys[i % len(api_keys)]
            agent = self.create_agent(i, api_key)
            agent.active_prob = active_probs[agent.id-1]
            agents[agent.id] = agent

        return agents


    def provider_agent_creation(self):
        """
        Create  provider  agents in parallel
        """
        agents = {}
        api_keys = list(self.config["api_keys"])
        agent_num = int(self.config["provider_agent_num"])
        # Add ONE user controllable user into the simulator if the flag is true.
        # We block the main thread when the user is creating the role.

        # if self.active_method == "random":
        #     active_probs = [self.config["active_prob"]] * agent_num
        # else:
        #     active_probs = np.random.pareto(self.config["active_prob"] * 10, agent_num)
        #     active_probs = active_probs / active_probs.max()

        for i in tqdm(range(1, agent_num+1)):
            api_key = api_keys[i % len(api_keys)]
            agent = self.create_provider_agent(i, api_key)
            agents[agent.id] = agent

        return agents

    def reset(self):
        # Reset the system
        self.pause()
        self.round_cnt = 0
        log_string = ""
        self.load_simulator()
        log_string = "The system is reset, and the historic records are removed."
        self.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
        return log_string



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
    wandb.init(
        # Set the project where this run will be logged
        project="basic-intro",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment",
        # Track hyperparameters and run metadata
        config=config)

    simulator.load_simulator()
    simulator.play()

    for step in tqdm(range(config['round']), desc='Round Processing...'):
        simulator.update_round(step)
        if simulator.config['provider_decision_making'] is False:
            print('No Decision')
        else:
            simulator.provider_round()

        reward = simulator.get_user_feedbacks()
        # rewards.extend(mini_batch_rewards)
        genre_count = simulator.get_genre_item_count()
        wandb.log({
            "total_rewards": reward,
            # "total_click": total_click,
            # **genre_count,
            # **ctr_feedbacks,
            # **provider_click_dict
        }
        )
    with open(simulator.config['item_save_path'], 'w') as json_file:
        json.dump(simulator.data.items, json_file)
        # if simulator.round_cnt == 1:
        #     df = pd.DataFrame([provider_click_dict])
        # else:
        #     df = pd.concat([df, pd.DataFrame([provider_click_dict])], ignore_index=True)
        # df.to_csv(f'/home/xiaopeng_ye/experiment/Agent4Fairness/figures/Bandwagon_effect/provider_dict.csv')
    wandb.finish()


