import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain_experimental.generative_agents.memory import (
    BaseMemory,
)
from langchain.prompts import PromptTemplate
from langchain_experimental.generative_agents import (
    GenerativeAgent,
)
from utils import utils
from utils.event import Event
from .recagent_memory import RecAgentMemory


class ProAgent(GenerativeAgent):
    id: int
    """The agent's unique identifier"""

    name: str
    """The agent's gender"""

    category_history: Dict = {}
    """The agent's traits"""

    items: Dict = {}
    """The agent's movie interest"""

    status: str = ''
    """The agent's action feature"""

    BUFFERSIZE: int = 10
    """The size of the agent's history buffer"""

    max_dialogue_token_limit: int = 600
    """The maximum number of tokens to use in a dialogue"""

    event: Event
    """The agent action"""

    active: bool = True
    """The probability of the agent being active"""

    no_action_round: int = 0
    """The number of rounds that the agent has not taken action"""

    memory: BaseMemory
    """The memory module in RecAgent."""

    role: str = "provider"

    item_round_exposure: list = []

    item_acc_exposure : dict = {}

    item_round_click: list = []

    item_acc_click : dict = {}

    profile_text : str = ""
    skill_belief : dict = {}
    audience_belief : dict = {}
    daily_comments : list = []
    new_round_item : list = []
    mode : str = "click_based"
    config : dict = {}
    active_prob: float = 1.0
    creation_state : np.ndarray = np.array([])
    def __init__(self, id, name, category_history, items, event, memory, memory_retriever, llm, config, active_prob):
        super().__init__(
            id=id,
            name=name,
            category_history=category_history,
            items=items,
            event=event,
            memory=memory,
            memory_retriever=memory_retriever,
            llm=llm,
            active_prob=active_prob
        )
        # self.item_round_exposure = []
        self.item_acc_exposure = {item: 0 for item in self.items}
        self.item_acc_click = {item: 0 for item in self.items}
        # self.tokenizer, m
        self.config = config
        self.creation_state = np.random.randn(len(utils.get_item_categories()))
        self.creation_state = utils.L2norm(self.creation_state)  # projection

    def upload_comments(self, comments):
        self.daily_comments.append(comments)


    def add_item(self, item_id, item_dict):
        self.new_round_item.append(item_id)
        self.items[item_id] = item_dict
        self.item_round_click[-1][item_id] = 0
        self.item_round_exposure[-1][item_id] = 0
        self.item_acc_exposure[item_id] = 0
        self.item_acc_click[item_id] = 0


    def __lt__(self, other: "ProAgent"):
        return self.event.end_time < other.event.end_time

    def update_exposure(self, item_id, round_cnt):

        if len(self.item_round_exposure) < round_cnt:
            self.item_round_exposure.append({item: 0 for item in self.items.keys()})

        self.item_round_exposure[round_cnt-1][item_id] += 1

        self.item_acc_exposure[item_id] += 1
        # print(f'item acc expo:{self.item_acc_exposure}')
        # print(f'item info:{self.items[item_id]}')

    def update_click(self, item_id, round_cnt):

        if len(self.item_round_click) < round_cnt:
            self.item_round_click.append({item: 0 for item in self.items.keys()})

        self.item_round_click[round_cnt-1][item_id] += 1
        self.item_acc_click[item_id] += 1
        # print(f'item acc click:{self.item_acc_click}')

    def update_round(self, round_cnt):

        # print(f'--- this is a new round for provider')
        # print(self.items)
        if len(self.item_round_click) < round_cnt:
            self.item_round_click.append({item: 0 for item in self.items.keys()})
        if len(self.item_round_exposure) < round_cnt:
            self.item_round_exposure.append({item: 0 for item in self.items.keys()})



    def reset_agent(self):
        """
        Reset the agent attributes, including memory, watched_history and heared_history.
        """
        # Remove watched_history and heared_history
        self.watched_history = []
        self.heared_history = []



    def _compute_agent_summary(self, observation) -> str:
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics about topic: {observation} given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(
                name=self.name,
                queries=[f"{self.name}'s core characteristics"],
                observation=observation,
            )
            .strip()
        )

    def get_summary(self, now, observation):
        """Return a descriptive summary of the agent."""
        prompt = PromptTemplate.from_template(
            "Given the following information about the content creator {agent_name}, please summarize the relevant details from his profile. His profile information is as follows:\n"
            + "Name: {agent_name}\n"
            # + "Genre Consistency: {agent_status}\n"
            + "Creating Genres History (<Genre>: <Create Times>): {history_category}\n"
            # + "Created Content History: {items}\n"
            + "Summary:"
        )
        kwargs: Dict[str, Any] = dict(
            observation=observation,
            agent_name=self.name,
            # agent_status='High' if self.status =='content' else 'Low',
            history_category=self.category_history,
            # items=item_names,
        )
        result = self.chain(prompt=prompt).run(**kwargs).strip()
        # print(f'summary:{result}')
        return (
            f"You are {self.name}, a content creator on the Youtube platform. \n{result}"
        )

    def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n"
            + suffix
        )
        now = datetime.now() if now is None else now
        agent_summary_description = self.get_summary(now=now, observation=observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            # agent_name=self.name,
            observation=observation,


        )

        # print(f'prompt:{prompt}')
        # print(kwargs)
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens

        # print(f'most recent memory :{consumed_tokens}')
        result = self.chain(prompt=prompt).run(**kwargs).strip()

        return result


    def get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        retriever = (
            self.memory.longTermMemory.memory_retriever
            if type(self.memory) == RecAgentMemory
            else self.memory.memory_retriever
        )
        result = []
        for doc in retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_dialogue_token_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_dialogue_token_limit:
                result.append(doc)
        if type(self.memory) == RecAgentMemory:
            result = self.memory.longTermMemory.format_memories_simple(result)
        else:
            result = self.memory.format_memories_simple(result)
        return result



    def construct_table(self, round_click_expo):
        df = pd.DataFrame(round_click_expo)
        # print(df)
        df.index = [f'round{index}' for index in df.index]

        name_row = [self.items[int(col)]['name'] for col in df.columns]
        name_row = pd.DataFrame([name_row], columns=df.columns)
        genre_row = [self.items[int(col)]['genre'] for col in df.columns]
        genre_row = pd.DataFrame([genre_row], columns=df.columns)

        df = pd.concat([name_row, genre_row, df])
        df = df.rename(index={df.index[0]: 'item name'})
        df = df.rename(index={df.index[1]: 'item genre'})
        df.columns = [f"item{str(col)}" for col in df.columns]
        exposure_table_str = df.to_markdown(index=True)
        return exposure_table_str

    def generate_exposure_click_str(self):
        round_click_expo = []
        for round_cnt in range(len(self.item_round_exposure)):
            # print(self.item_round_exposure)
            exp_dic = self.item_round_exposure[round_cnt]
            # print(self.item_round_click)
            # print(round_cnt)
            click_dic = self.item_round_click[round_cnt]
            # print(f'click dict keys{click_dic.keys()}')
            # print(f'data keys :{self.items.keys()}')
            round_click_expo.append({i : f'{click_dic[i]}/{exp_dic[i]}' for i in click_dic.keys()})
        return self.construct_table(round_click_expo)


    def get_user_comments_str(self, topk):
        user_comments = ''
        for i, c in enumerate(self.daily_comments[-topk:]):
            user_comments = user_comments + f'{i}.' + c + '\n'
        return user_comments

    def get_item_click(self, new_item_id):
        item_create_times = 0
        item_clicks = 0
        for round_click_dict in self.item_round_click[:-1]:
            for item_id, item_click in round_click_dict.items():
                if item_id == new_item_id:
                    item_create_times += 1
                    item_clicks += item_click
        return "{:.3f}".format(item_clicks) if item_create_times!= 0 else "0.0"



    def get_genre_click_ctr(self, category_list):

        audience_belief = {cate: 0 for cate in category_list}
        genre_click = {cate: 0 for cate in category_list}
        genre_expo = {cate: 0 for cate in category_list}
        genre_avg_click_per_item = {cate: 0 for cate in category_list}
        # audience belief is the total CTR of each genre.
        for item_id, item_click in self.item_acc_click.items():
            genre = self.items[item_id]['genre']
            genre_click[genre] += item_click

        for item_id, item_expo in self.item_acc_exposure.items():
            genre = self.items[item_id]['genre']
            genre_expo[genre] += item_expo

        genre_items = {cate: 0 for cate in category_list}

        # for round_click_dict in self.item_round_click[:-1]:
        #     for item_id, item_click in round_click_dict.items():
        #         genre = self.items[item_id]['genre']
        #         genre_items[genre] += 1
        for item_id in self.new_round_item:
            genre = self.items[item_id]['genre']
            genre_items[genre] += 1


        for cate in category_list:
            if genre_items[cate] == 0:
                audience_belief[cate] = 'unknown'
                genre_avg_click_per_item[cate] = 'unknown'
            else:
                audience_belief[cate] = "{:.3f}".format(genre_click[cate] / genre_expo[cate]) if genre_expo[cate]!= 0 else "0.0"  # CTR
                genre_avg_click_per_item[cate] = "{:.3f}".format(genre_click[cate] / genre_items[cate])
            # sum_ctr += audience_belief[cate]
        # for cate in category_list:
        #     audience_belief[cate] = audience_belief[cate]/sum_ctr if sum_ctr != 0 else 0
        return audience_belief, genre_click, genre_expo, genre_avg_click_per_item

    def find_first_match(self, text, words_list):
        pattern = '|'.join(re.escape(word) for word in words_list)
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def filter_lines_with_colon(self, text):
        # 将文本按行分割
        lines = text.split('\n')

        # 筛选包含 '::' 的行
        for line in lines:
            if '::' in line:
                return line

        # 将筛选后的行合并成一个字符串
        return ''

    def get_max_genre(self, categories):
        cates_ctr, cates_click, cates_expo, cates_avg_click = self.get_genre_click_ctr(category_list=categories)
        print(f'cates_avg_click:{cates_avg_click}')
        max_genre = 'unknown'
        max_click = 0.0
        for genre, avg_click in cates_avg_click.items():
            if avg_click == 'unknown':
                continue
            else:
                avg_click_num = float(avg_click)
                if avg_click_num >= max_click:
                    max_click = avg_click_num
                    max_genre = genre
        return max_genre


    def get_analyze_prompt(self, categories, round_cnt, user_interest_dict=None):
        cates_ctr, cates_click, cates_expo, cates_avg_click = self.get_genre_click_ctr(category_list=categories)
        # print(f'provider round ctr: {cates_ctr}')
        known_cates = []
        unknown_cates = []
        for cate, ctr in cates_ctr.items():
            if ctr =='unknown':
                unknown_cates.append(cate)
            else:
                known_cates.append(cate)
        new_item_id, new_item_genre, new_item_click, new_item_expo, new_item_ctr = None, None,None,None,None

        if self.config['provider_decision_policy'] == 'full_information':
            if len(self.new_round_item) > 0:
                new_item_id = self.new_round_item[-1]   # 现在还没创作item，上一个创作的item就是-1
                new_item_genre = self.items[new_item_id]['genre']
                new_item_click = self.item_acc_click[new_item_id]  #self.item_round_click[-2][new_item_id]   # 现在已经更新了item_round_click， last round是-2
                new_item_expo = self.item_acc_exposure[new_item_id] #self.item_round_exposure[-2][new_item_id]  #  last round
                new_item_ctr = "{:.4f}".format(new_item_click / new_item_expo) if new_item_expo != 0 else 'unknown'
                suffix = (
                    f"The distribution of users interested in each genre is as below:\n{{{user_interest_dict}}}."
                    f"The average clicks per item of each genre {self.name} has created is as below:\n{{{cates_avg_click}}}. ([unknown] means the item genre {self.name} have not explored.)"
                    f"\nRecently, {self.name} create a item of genre [{new_item_genre}], and receives {new_item_click} clicks." 
                    f"\nDue to the statistical data, {self.name}'s expertise and creation consistency, {self.name} must choose one genre to create to gain more user attention:"
                          f"\nWrite [CREATE]:: <genre_name> (e.g., [CREATE]:: Music)."
                          f"\nVideo genre should be chosen from {categories}"
                          "\nPlease answer concisely and strictly follow the output rules."
                        )
            else:
                suffix = (
                    f"The distribution of users interested in each genre is as below:\n{{{user_interest_dict}}}."
                    f"\nDue to the statistical data, {self.name}'s expertise and creation consistency, {self.name} must choose one genre to create to gain more user attention:"
                          f"\nWrite [CREATE]:: <genre_name> (e.g., [CREATE]:: Music)."
                          f"\nVideo genre should be chosen from {categories}"
                          "\nPlease answer concisely and strictly follow the output rules."
                        )
        else:
            if len(self.new_round_item) > 0:
                new_item_id = self.new_round_item[-1]   # 现在还没创作item，上一个创作的item就是-1
                new_item_genre = self.items[new_item_id]['genre']
                new_item_click = self.item_acc_click[new_item_id]  #self.item_round_click[-2][new_item_id]   # 现在已经更新了item_round_click， last round是-2
                new_item_expo = self.item_acc_exposure[new_item_id] #self.item_round_exposure[-2][new_item_id]  #  last round
                new_item_ctr = "{:.4f}".format(new_item_click / new_item_expo) if new_item_expo != 0 else 'unknown'
                if self.config['provider_decision_mode'] == 'CTR_based':
                    suffix = (f"The CTR (Click-Through Rate) of each genre {self.name} has created is as below:\n{{{cates_ctr}}}. ([unknown] means the item genre {self.name} have not explored.)"
                          f"\nRecently, {self.name} create a item of genre [{new_item_genre}], and receives a CTR of {new_item_ctr} ({new_item_click} clicks out of {new_item_expo} exposure.)" 
                          f"\nDue to the statistical data, {self.name}'s expertise and creation consistency, {self.name} must choose one of the two actions below to obtain more user clicks:"
                         f"\n (1) [EXPLORE] Create content in a new genre that has not been explored before, which means other genres may have a larger audience and more opportunities to profit. But it might not be {self.name}'s area of expertise and requires greater effort to create."  # 
                         f"\n (2) [EXPLOIT] Sticking to creating content of familiar genre, which means {self.name} will leverage his creative expertise building a stable brand identity. But it might limit {self.name}'s audience reach and lead to insufficient income."  #But it might limit {self.name}'s audience   #However, persistently creating niche genre may lead to insufficient income
                          f"\nTo explore a new genre, write: [EXPLORE]:: <genre_name>. If so, give the specific genre name chosen from {unknown_cates}."
                            f"\nTo sticking to familiar genres, write: [EXPLOIT]:: <genre_name>. If so, give the specific genre name chosen from {known_cates}."
                          "\nPlease answer concisely and strictly follow the output rules."
                          )
                elif self.config['provider_decision_mode'] == 'click_based':
                    suffix = (f"The average clicks per item of each genre <{self.name}> has created is as below:\n{{{cates_avg_click}}}. ([unknown] means the item genre <{self.name}> have not explored.)"
                          f"\nRecently, <{self.name}> create a item of genre [{new_item_genre}], and receives {new_item_click} clicks." 
                          f"\nDue to the statistical data, <{self.name}>'s expertise and creation consistency, {self.name} must choose one of the two actions below to obtain more user clicks:"
                         f"\n (1) [EXPLORE] Create content in a new genre that has not been explored before, which means other genres may have a larger audience and more opportunities to profit. But it might not be <{self.name}>'s area of expertise and requires greater effort to create."  # 
                         f"\n (2) [EXPLOIT] Sticking to creating content of familiar genre, which means {self.name} will leverage his creative expertise building a stable brand identity. But it might limit <{self.name}>'s audience reach and lead to insufficient income."  #But it might limit {self.name}'s audience   #However, persistently creating niche genre may lead to insufficient income
                          f"\nTo explore a new genre, write: [EXPLORE]:: <genre_name>. If so, give the specific genre name chosen from {unknown_cates}."
                            f"\nTo sticking to familiar genres, write: [EXPLOIT]:: <genre_name>. If so, give the specific genre name chosen from {known_cates}."
                          "\nPlease answer concisely and strictly follow the output rules."
                          )
                    with open(f'/home/xiaopeng_ye/experiment/Agent4Fairness/figures/prospect_theory/DIN_{self.config["provider_decision_policy"]}_{self.config["reranking_model"]}_item_recency_{self.config["item_recency"]}_record.txt', 'a') as f:
                        f.write( f'{round_cnt}\t{self.name}\t{new_item_click}\n')  # 追加内容，不会换行
                else:
                    raise ValueError(f'Unvalid mode found:{self.mode}')
            else:
                suffix = (
                      f"Due to {self.name}'s expertise, {self.name} must choose one genre to create to gain more user attention:"
                      f"\nWrite [CREATE]:: <genre_name> (e.g., [CREATE]:: Music)."
                      f"\nVideo genre should be chosen from {categories}"
                      "\nPlease answer concisely and strictly follow the output rules."
                    )

        return suffix

    def analyzing(self, categories, round_cnt,user_interest_dict=None ): #-> Tuple[str, str]:

        cates_ctr, cates_click, cates_expo, cates_avg_click = self.get_genre_click_ctr(category_list=categories)
        # print(f'provider round ctr: {cates_ctr}')
        known_cates = []
        unknown_cates = []
        for cate, ctr in cates_ctr.items():
            if ctr =='unknown':
                unknown_cates.append(cate)
            else:
                known_cates.append(cate)
        new_item_id, new_item_genre, new_item_click, new_item_expo, new_item_ctr = None, None,None,None,None

        if self.config['provider_decision_policy'] == 'full_information':
            if len(self.new_round_item) > 0:
                new_item_id = self.new_round_item[-1]   # 现在还没创作item，上一个创作的item就是-1
                new_item_genre = self.items[new_item_id]['genre']
                new_item_click = self.item_acc_click[new_item_id]  #self.item_round_click[-2][new_item_id]   # 现在已经更新了item_round_click， last round是-2
                new_item_expo = self.item_acc_exposure[new_item_id] #self.item_round_exposure[-2][new_item_id]  #  last round
                new_item_ctr = "{:.4f}".format(new_item_click / new_item_expo) if new_item_expo != 0 else 'unknown'
                suffix = (
                    f"The distribution of users interested in each genre is as below:\n{{{user_interest_dict}}}."
                    f"The average clicks per item of each genre {self.name} has created is as below:\n{{{cates_avg_click}}}. ([unknown] means the item genre {self.name} have not explored.)"
                    f"\nRecently, {self.name} create a item of genre [{new_item_genre}], and receives {new_item_click} clicks." 
                    f"\nDue to the statistical data, {self.name}'s expertise and creation consistency, {self.name} must choose one genre to create to gain more user attention:"
                          f"\nWrite [CREATE]:: <genre_name> (e.g., [CREATE]:: Music)."
                          f"\nVideo genre should be chosen from {categories}"
                          "\nPlease answer concisely and strictly follow the output rules."
                        )
            else:
                suffix = (
                    f"The distribution of users interested in each genre is as below:\n{{{user_interest_dict}}}."
                    f"\nDue to the statistical data, {self.name}'s expertise and creation consistency, {self.name} must choose one genre to create to gain more user attention:"
                          f"\nWrite [CREATE]:: <genre_name> (e.g., [CREATE]:: Music)."
                          f"\nVideo genre should be chosen from {categories}"
                          "\nPlease answer concisely and strictly follow the output rules."
                        )
        else:
            if len(self.new_round_item) > 0:
                new_item_id = self.new_round_item[-1]   # 现在还没创作item，上一个创作的item就是-1
                new_item_genre = self.items[new_item_id]['genre']
                new_item_click = self.item_acc_click[new_item_id]  #self.item_round_click[-2][new_item_id]   # 现在已经更新了item_round_click， last round是-2
                new_item_expo = self.item_acc_exposure[new_item_id] #self.item_round_exposure[-2][new_item_id]  #  last round
                new_item_ctr = "{:.4f}".format(new_item_click / new_item_expo) if new_item_expo != 0 else 'unknown'
                if self.config['provider_decision_mode'] == 'CTR_based':
                    suffix = (f"The CTR (Click-Through Rate) of each genre {self.name} has created is as below:\n{{{cates_ctr}}}. ([unknown] means the item genre {self.name} have not explored.)"
                          f"\nRecently, {self.name} create a item of genre [{new_item_genre}], and receives a CTR of {new_item_ctr} ({new_item_click} clicks out of {new_item_expo} exposure.)" 
                          f"\nDue to the statistical data, {self.name}'s expertise and creation consistency, {self.name} must choose one of the two actions below to obtain more user clicks:"
                         f"\n (1) [EXPLORE] Create content in a new genre that has not been explored before, which means other genres may have a larger audience and more opportunities to profit. But it might not be {self.name}'s area of expertise and requires greater effort to create."  # 
                         f"\n (2) [EXPLOIT] Sticking to creating content of familiar genre, which means {self.name} will leverage his creative expertise building a stable brand identity. But it might limit {self.name}'s audience reach and lead to insufficient income."  #But it might limit {self.name}'s audience   #However, persistently creating niche genre may lead to insufficient income
                          f"\nTo explore a new genre, write: [EXPLORE]:: <genre_name>. If so, give the specific genre name chosen from {unknown_cates}."
                            f"\nTo sticking to familiar genres, write: [EXPLOIT]:: <genre_name>. If so, give the specific genre name chosen from {known_cates}."
                          "\nPlease answer concisely and strictly follow the output rules."
                          )
                elif self.config['provider_decision_mode'] == 'click_based':
                    suffix = (f"The average clicks per item of each genre <{self.name}> has created is as below:\n{{{cates_avg_click}}}. ([unknown] means the item genre <{self.name}> have not explored.)"
                          f"\nRecently, <{self.name}> create a item of genre [{new_item_genre}], and receives {new_item_click} clicks." 
                          f"\nDue to the statistical data, <{self.name}>'s expertise and creation consistency, {self.name} must choose one of the two actions below to obtain more user clicks:"
                         f"\n (1) [EXPLORE] Create content in a new genre that has not been explored before, which means other genres may have a larger audience and more opportunities to profit. But it might not be <{self.name}>'s area of expertise and requires greater effort to create."  # 
                         f"\n (2) [EXPLOIT] Sticking to creating content of familiar genre, which means {self.name} will leverage his creative expertise building a stable brand identity. But it might limit <{self.name}>'s audience reach and lead to insufficient income."  #But it might limit {self.name}'s audience   #However, persistently creating niche genre may lead to insufficient income
                          f"\nTo explore a new genre, write: [EXPLORE]:: <genre_name>. If so, give the specific genre name chosen from {unknown_cates}."
                            f"\nTo sticking to familiar genres, write: [EXPLOIT]:: <genre_name>. If so, give the specific genre name chosen from {known_cates}."
                          "\nPlease answer concisely and strictly follow the output rules."
                          )
                else:
                    raise ValueError(f'Unvalid mode found:{self.mode}')
            else:
                suffix = (
                      f"Due to {self.name}'s expertise, {self.name} must choose one genre to create to gain more user attention:"
                      f"\nWrite [CREATE]:: <genre_name> (e.g., [CREATE]:: Music)."
                      f"\nVideo genre should be chosen from {categories}"
                      "\nPlease answer concisely and strictly follow the output rules."
                    )

        prompt = suffix
        # print(f'analyzing prompt:{prompt}')
        response = self.llm(prompt=prompt, profile=self.profile_text)
        # print(f'analyzing response:{response}')
        return prompt, response

    def get_recent_creation(self, category):
        new_item_dict = None
        feature_list = ['name', 'genre', 'tags', 'description']
        chosen_itemid = None

        cate_item_dict = {}
        for item_id in self.items.keys():
            item_genre = self.items[item_id]['genre']
            item_description_len = self.items[item_id]['description']
            if item_genre == category:
                cate_item_dict[item_id] = item_description_len

        if len(cate_item_dict) == 0:
            return None
        else:
            min_itemid = min(cate_item_dict, key=cate_item_dict.get)
            new_item_dict = {k: self.items[min_itemid][k] for k in feature_list}
            return new_item_dict

        # if len(self.new_round_item) > 0:
        #     for item_id in self.new_round_item[::-1]:
        #         item_genre = self.items[item_id]['genre']
        #         if item_genre == category:
        #             new_item_dict = {k: self.items[item_id][k] for k in feature_list}
        #             return new_item_dict
        #         else:
        #             continue


    def creating(self, now, conclusion, categories):
        history_items = self.items.values()
        if len(history_items) > 3:
            import random
            # print(type(history_items))
            history_items = random.sample(list(history_items), 3)

        history_str = ""
        for item in history_items:
            need_item_feature = ['name', 'genre', 'tags', 'description']
            sub_dict = {key: item[key] for key in need_item_feature}
            history_str += "{" + str(sub_dict) + "}\n"
        suffix = (f"Based on the analysis :{conclusion}, please create ONE new content for {self.name} that fit user's interest."
                  + f"\nYou can refer to the creation history of {self.name}: {history_str}"
                  + '''\nIMPORTANT NOTICE: Response in JSON dictionary format. Write {{"name": [item_name], "genre": Genre1|Genre2|....,  "tags": [tag1, tag2, tag3], "description":"item_description_text"}} ''')

        observation = f"Item genre should be chosen from {categories}. Response should be in JSON dictionary format."

        result = self._generate_reaction(observation, suffix, now)
        # dict_pattern = r"{.*}"
        import ast
        # print(f'---{result} ---')
        start_index = result.find("{")
        end_index = result.find("}")
        dict_result = result[start_index: end_index+1]
        # print(f'---{dict_result} ---')
        # result = re.findall(dict_pattern, result)
        # print(f'---222{result} ---')
        try:
            dict_result = ast.literal_eval(dict_result)
        except SyntaxError:
            # prompt = PromptTemplate.from_template(template='''{observation}Convert the following content into JSON dictionary format:{input_text}. Write: {{"name": [item_name], "genre": Genre1|Genre2|....,  "tags": [tag1, tag2, tag3], "description":"item_description_text"}} ''')
            #
            # kwargs: Dict[str, Any] = dict(
            #     input_text=result,
            #     observation='',
            # )
            prompt = f"Convert the following content into JSON dictionary format:{result}. " + '''Write: \{{"name": [item_name], "genre": Genre1|Genre2|....,  "tags": [tag1, tag2, tag3], "description":"item_description_text"}} '''
            temp_result = self.llm(prompt) #self.chain(prompt=prompt).run(**kwargs).strip()
            # print(f'---{dict_result} ---')
            start_index = temp_result.find("{")
            end_index = temp_result.find("}")
            dict_result = temp_result[start_index: end_index + 1]
            dict_result = ast.literal_eval(dict_result)

        # print(f'---{dict_result} ---')
        item_name = dict_result['name']
        item_genre = dict_result['genre']
        item_description = dict_result['description']
        # self.memory.save_context(
        #     {},
        #     {
        #         self.memory.add_memory_key: f"{self.name} create content: {result}",
        #         self.memory.now_key: now,
        #     },
        # )


        return item_name, item_genre, item_description

    def analyse_status(self, now, round_cnt, profile) -> Tuple[str, str]:

        exposure_table_str = self.generate_exposure_click_str()
        # print(f'exposure str:\n{exposure_table_str}')
        call_to_action_template = (
                f"Here is the click/exposure data for each item {self.name} created in previous round:\n{exposure_table_str}. Please rate the operational status from 1 to 10 according to the following scoring rule."
                f"\nCrisis Status (1): {self.name}'s content receives minimal interaction, with very few clicks, and exposures consistently."
                f"\nWarning Status(5): {self.name} The number of clicks and exposures on content decreases recently, with reduced user interaction."
                f"\nHealthy Status(10): {self.name} receives a high number of click and exposures, with active fan interaction."
                f"Based on the status. What action would {self.name} like to take? Respond in one line."
                f"\nIf {self.name} wants to create content on the platform, write:\n [CREATE]:: {self.name} choose to stay in the platform to create content."
                # + "\nIf {agent_name} wants to stop creating content, write:\n [STOP]:: {agent_name} stops creating content"
                f"\nIf {self.name} wants to quit the platform, write:\n [QUIT]:: {self.name} quits the platform"
        )
        observation = (f"{self.name} is in round {round_cnt}."
            f"{self.name} must take only ONE of the actions below:(1) Create Content. If so, {self.name} will endeavour to create content on the platform to make profit."
            # f"\n(2) Stop Creating Content. {self.name} can can stop working for a while. If {self.name} found that creating too often or otherwise."
            f"\n(2) Quit. If so, {self.name} will quit the platform and transfer to another platform to seek greater profits.")

        # full_result = self._generate_reaction(observation, call_to_action_template, now)
        prompt = (f"Observation: {observation}"
                 f"\n {call_to_action_template}")
        full_result = self.llm(prompt, profile=profile)
        # print(f'full result:{full_result}')
        # result = full_result.strip().split("\n")[0]

        # choice = full_result.split("::")[0]
        # action = result.split("::")[1]

        # self.memory.save_context(
        #     {},
        #     {
        #         self.memory.add_memory_key: f"{self.name} take action: " f"{full_result}",
        #         self.memory.now_key: now,
        #     },
        # )
        q_r_pair = [prompt, full_result]
        return q_r_pair


    def initialize_provider_profile(self, profile=None):
        if profile is None:
            prompt = (
                f"Given the following information about the content creator {self.name} on Youtube:\n"
                + f"Name: {self.name}\n"
                + f"Genre Creation History (<genre_name>: <creation times>): {self.category_history}"
                # + f"Genre Consistency: {'High' if self.status =='content' else 'Low'}\n"
                # + f"Created Content History: {self.items}\n"
                # + "Summary:"
                + f"Based on the information above, please summarize the basic characteristics of the content creator <{self.name}>(in the second person), including its name, the categories <{self.name}> excels in creating, and <{self.name}>'s level of focus—whether <{self.name}> concentrates on one genre or actively explore multiple genres."
                # + f"\nWrite an summary about the information of <{self.name}> ( in the second person ), "
                + f"\nWrite 'You are a content creator on Youtube and your nickname is {self.name} ...'"
            )
            result = self.llm(prompt, history=[], profile='') #self.chain(prompt=prompt).run(**kwargs).strip()

            cleaned_text = re.sub(r'\n?Here is the summary:\n?', '', result)
            cleaned_text = re.sub(r"\n?Here's the summary:\n?", '', cleaned_text)
            print(f'profile:{cleaned_text}')
            self.profile_text = cleaned_text + 'You need to do your best to get more user clicks.'
        else:
            self.profile_text = profile
        return self.profile_text
    def get_profile_text(self):
        return self.profile_text



    def extract_analyze_result(self, categories, result):
        if len(self.new_round_item) > 0 and self.config['provider_decision_policy'] == 'bounded_rational':
            new_item_id = self.new_round_item[-1]
            new_item_genre = self.items[new_item_id]['genre']

            if result.find("[EXPLORE]") != -1 or result.find("EXPLORE") != -1:
                choice = "[EXPLORE]"
                # temp_choice, action, *rest = result.split("::")

                genre = self.find_first_match(result, categories)

                action = f"{self.name} chooses to explore a new genre {genre}."
            else:
                choice = "[EXPLOIT]"
                genre = self.find_first_match(result, categories)  #new_item_genre #
                action = f"{self.name} chooses to stick to creating content of the {genre} genre."

            if genre == None:
                genre = new_item_genre
                action = f"{self.name} chooses to create video of genre {genre}."
        else:
            choice = "[CREATE]"
            # temp_choice, action, *rest = result.split("::")

            genre = self.find_first_match(result, categories)
            action = f"{self.name} chooses to create video of genre {genre}."
            if genre == None:

                genre = random.choice(categories)
                action = f"{self.name} chooses to create video of genre {genre}."

        # print(f'provider decision:{choice}:: {genre}')
        return genre, choice, action

    def get_active_prob(self, method) -> float:
        if method == "marginal":
            return self.active_prob * (self.no_action_round + 1)
        else:
            return self.active_prob

    # def get_response_fromLLM(self, prompt, history, profile):
    #     # tokenizer = AutoTokenizer.from_pretrained(
    #     #     '/home/xiaopeng_ye/LLMs/Meta-Llama-3-8B-Instruct', trust_remote_code=True)
    #     # model = AutoModelForCausalLM.from_pretrained(
    #     #     '/home/xiaopeng_ye/LLMs/Meta-Llama-3-8B-Instruct',
    #     #     torch_dtype=torch.float16,
    #     #     device_map='auto',  # "sequential",
    #     #     trust_remote_code=True
    #     # )
    #     # model.half()
    #     # tokenizer.half()
    #
    def generating(self, action, choice, choose_genre, analyze_history):
        recent_creation = self.get_recent_creation(category=choose_genre)
        prompt = (f"Based on the analysis, {action}"
                    f"\n Please create a brand new item in the {choose_genre} genre."
                    "\n Return the results strictly according to the following JSON dictionary format: \n"
                    '''\nWrite: {"name": "item_name", "genre": "''' + choose_genre + '''", "tags": [tag1, tag2, tag3], "description": "item_description_text"}'''
                  )
        prompt = prompt + f"\nYou can draw inspiration from {self.name}'s previous creation on genre {choose_genre}, but cannot replicate them identically.\n{recent_creation}" if (
                                        recent_creation != None and choice != '[RANDOM]') else ""
        prompt = prompt + "\nPlease answer concisely and strictly follow the output rules:" + '''\n {"name": "item_name", "genre": "''' + choose_genre + '''", "tags": [tag1, tag2, tag3], "description": "item_description_text"}'''
        response = self.llm(prompt=prompt, profile=self.profile_text, history=analyze_history)
        return response


    def get_creation_utility_vector(self):  # users utility
        categories = utils.get_item_categories()
        cates_ctr, cates_click, cates_expo, cates_avg_click = self.get_genre_click_ctr(category_list=categories)
        creation_utility_vector = []
        for cate_name in categories:
            c_avg_click = cates_avg_click[cate_name]
            if c_avg_click == 'unknown':
                creation_utility_vector.append(0.0)
            else:
                creation_utility_vector.append(float(c_avg_click))
        return np.array(creation_utility_vector)

    def get_creation_state(self):
        return self.creation_state