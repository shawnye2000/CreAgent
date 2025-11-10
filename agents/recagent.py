import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain_experimental.generative_agents.memory import (
    GenerativeAgentMemory,
    BaseMemory,
)
from langchain.prompts import PromptTemplate
from langchain_experimental.generative_agents import (
    GenerativeAgent,
)
from utils.event import Event
from .recagent_memory import RecAgentMemory


class RecAgent(GenerativeAgent):
    id: int
    """The agent's unique identifier"""

    gender: str = ''
    """The agent's gender"""

    traits: str = ''
    """The agent's traits"""

    status: str = ''

    interest: str
    """The agent's movie interest"""
    #
    feature: str = ''
    # """The agent's action feature"""

    relationships: Dict[str, str] = {}
    """The agent's relationship with other agents"""

    watched_history: List[str] = []
    """The agent's history of watched movies"""

    heared_history: List[str] = []
    """The agent's history of heared movies"""

    BUFFERSIZE: int = 10
    """The size of the agent's history buffer"""

    max_dialogue_token_limit: int = 600
    """The maximum number of tokens to use in a dialogue"""

    event: Event
    """The agent action"""

    active_prob: float = 0.5
    """The probability of the agent being active"""

    no_action_round: int = 0
    """The number of rounds that the agent has not taken action"""

    memory: BaseMemory
    """The memory module in RecAgent."""

    role: str = "agent"


    def __lt__(self, other: "RecAgent"):
        return self.event.end_time < other.event.end_time

    def get_active_prob(self, method) -> float:
        if method == "marginal":
            return self.active_prob * (self.no_action_round + 1)
        else:
            return self.active_prob

    def update_from_dict(self, data_dict: dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    def interact_agent(self):
        """
        type the sentences you want to interact with the agent.
        """

        interact_sentence = input(
            "Please type the sentence you want to interact with {}: ".format(self.name)
        )

        result = self.interact_reaction(interact_sentence)[1]
        return interact_sentence, result

    def modify_agent(self):
        """
        modify the attribute of agent, including age, traits, status
        """
        age = input(
            "If you want to modify the age, please enter the information. Otherwise, enter 'n' to skip it: "
        )
        gender = input(
            "If you want to modify the gender, please enter the information. Otherwise, enter 'n' to skip it: "
        )
        traits = input(
            "If you want to modify the traits, please enter the information. Otherwise, enter 'n' to skip it: "
        )
        status = input(
            "If you want to modify the status, please enter the information. Otherwise, enter 'n' to skip it: "
        )

        self.age = age if age not in "n" else self.age
        self.gender = gender if gender not in "n" else self.gender
        self.traits = traits if traits not in "n" else self.traits
        self.status = status if status not in "n" else self.status

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

    def get_summary(
        self,
        now: Optional[datetime] = None,
        observation: str = None,
    ) -> str:
        """Return a descriptive summary of the agent."""
        prompt = PromptTemplate.from_template(
            "Given the following observation about {agent_name}: '{observation}', please summarize the relevant details from his profile. His profile information is as follows:\n"
            + "Name: {agent_name}\n"
            # + "Age: {agent_age}\n"
            # + "Gender:{agent_gender}\n"
            # + "Traits: {agent_traits}\n"
            # + "Status: {agent_status}\n"
            + "Video Genre Interest: {agent_interest}\n"
            # + "Click History by Genre (<genre_name>: <click count>): {category_history}"
            # + "Feature: {agent_feature}\n"
            # + "Interpersonal Relationships: {agent_relationships}\n"
            + "Please avoid repeating the observation in the summary.\nSummary:"
        )
        kwargs: Dict[str, Any] = dict(
            observation=observation,
            agent_name=self.name,
            # agent_age=self.age,
            # agent_gender=self.gender,
            # agent_traits=self.traits,
            # agent_status=self.status,
            agent_interest=self.interest,
            # category_history= self.category_history
            # agent_feature=self.feature,
            # agent_relationships=self.relationships,
        )
        # print(f'user summary:{prompt}')
        result = self.chain(prompt=prompt).run(**kwargs).strip()
        # age = self.age if self.age is not None else "N/A"
        return (
            f"Name: {self.name} Video Interest: {self.interest}" + f"\n{result}"
        )

    def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            # + "\nIt is {current_time}."
            + "\n{agent_name}'s video interest is in {video_interest}."
            # + "\n{agent_name} recently watched {watched_history} on recommender system."
            # + "\nOther than that {agent_name} doesn't know any videos."
            # + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            # + "\nAll occurrences of video names should be enclosed with <>"
            + "\n\n"
            + suffix
        )
        now = datetime.now() if now is None else now
        agent_summary_description = self.get_summary(now=now, observation=observation)
        # current_time_str = (
        #     datetime.now().strftime("%B %d, %Y, %I:%M %p")
        #     if now is None
        #     else now.strftime("%B %d, %Y, %I:%M %p")
        # )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            # current_time=current_time_str,
            video_interest=self.interest,
            agent_name=self.name,
            observation=observation,
            # watched_history=(
            #     self.watched_history if len(self.watched_history) > 0 else "nothing"
            # ),
            # heared_history=(
            #     self.heared_history if len(self.heared_history) > 0 else "nothing"
            # ),
        )
        # consumed_tokens = self.llm.get_num_tokens(
        #     prompt.format(most_recent_memories="", **kwargs)
        # )
        # kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        # print(f'most_recent_memory:{consumed_tokens}')
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


    def take_action(self, now) -> Tuple[str, str]:
        """Take one of the actions below.
        (1) Enter the Recommender.
        (2) Do Nothing.
        """
        call_to_action_template = (
            "What action would {agent_name} like to take? Respond in one line."
            + "\nIf {agent_name} wants to enter the Recommender System, write:\n [RECOMMENDER]:: {agent_name} enters the Recommender System"
            # + "\nIf {agent_name} wants to enter the Social Media, write:\n [SOCIAL]:: {agent_name} enters the Social Media"
            + "\nIf {agent_name} wants to do nothing, write:\n [NOTHING]:: {agent_name} does nothing"
        )
        observation = f"{self.name} must take only ONE of the actions below:(1) Enter the Recommender System. If so, {self.name} will be recommended some videos, from which {self.name} can watch some videos.\n(1) Do Nothing."  # or search for movies by himself
        full_result = self._generate_reaction(observation, call_to_action_template, now)
        result = full_result.strip().split("\n")[0]

        choice = result.split("::")[0]
        # action = result.split("::")[1]

        # self.memory.save_context(
        #     {},
        #     {
        #         self.memory.add_memory_key: f"{self.name} take action: " f"{result}",
        #         self.memory.now_key: now,
        #     },
        # )
        return choice, result

    def take_recommender_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the four actions below.
        (1) Watch movies among the recommended items.
        (2) Next page.
        (3) Search items.
        (4) Leave the recommender system.
        """
        call_to_action_template = (
            "{agent_name} must choose one of the four actions below:\n"
            "(1) Watch ONLY ONE video from the list returned by the recommender system.\n"
            "(2) See the next page.\n"
            "(3) Leave the recommender system."
            + "\nTo watch a video from the recommended list that match {agent_name}'s interests, write:\n[WATCH]:: Index of the video starting from 1 (e.g., [WATCH]:: 3)"
            # + "\nTo see the next page, write:\n[NEXT]:: {agent_name} views the next page."
            # + "\nTo leave the recommender system, write:\n[LEAVE]:: {agent_name} leaves the recommender system."
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = full_result.strip()
        # substring = 'must choose one of the four actions below'
        # index = result.find(substring)
        # if index != -1:
        #     # 使用replace方法删除子字符串
        #     result = result[:index-2]
        # print(f'-----------------------------{result}')
        if result.find("::") != -1:

            choice, action, *rest = result.split("::")
            choice = choice.strip()
            match = re.search(r'(\d+)', action)

            if match:
                num = int(match.group(1))
                if 1 <= num <= 5:
                    action = num
                else:
                    action = 1

        else:
            choice = "[LEAVE]"
            action = f"{self.name} leaves the recommender system."
        # self.memory.save_context(
        #     {},
        #     {
        #         self.memory.add_memory_key: f"{self.name} took action: {result}",
        #         self.memory.now_key: now,
        #     },
        # )

        return choice, action


    def take_click_action_with_rec(self, observation, now) -> Tuple[str, str]:
        """Take one of the four actions below.
        (1) Watch movies among the recommended items.
        (2) Next page.
        (3) Search items.
        (4) Leave the recommender system.
        """
        call_to_action_template = (
             "{agent_name} is an extremely cautious user who is very particular about what you interact with. {agent_name} will not click on any item unless he/she is absolutely sure he/she likes it. If there is even the slightest hesitation or doubt, {agent_name} will refrain from watching.\n"
            "{agent_name} must choose one of the two actions below:\n"
            "(1) Skip the current video.\n"
            "(2) Watch the video recommended by the recommender system.\n"
            "(3) Leave the recommender system."
            + "\nTo skip the current recommended video since {agent_name} don't want to waste time on it and prefers to watch other videos that attract him more, write:\n[SKIP] (e.g., [SKIP])."
            + "\nTo watch the current recommended video since it matches {agent_name}'s interests, write:\n[WATCH] (e.g., [WATCH])"
            + "\nTo leave the recommender system, write:\n[LEAVE] (e.g., [LEAVE])."
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = full_result.strip()
        # substring = 'must choose one of the four actions below'
        # index = result.find(substring)
        # if index != -1:
        #     # 使用replace方法删除子字符串
        #     result = result[:index-2]
        # print(f'-----------------------------{result}')

        if result.find("[WATCH]") != -1:
            choice = "[WATCH]"
            action = f"{self.name} watches the current video."
            # self.memory.save_context(
            #     {},
            #     {
            #         self.memory.add_memory_key: f"{self.name} took action: {result}",
            #         self.memory.now_key: now,
            #     },
            # )
        elif result.find("[SKIP]") != -1:
            choice = "[SKIP]"
            action = f"{self.name} skip the current video."
        else:
            choice = "[LEAVE]"
            action = f"{self.name} leaves the recommender system."

        return choice, action
    def take_click_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the four actions below.
        (1) Watch movies among the recommended items.
        (2) Next page.
        (3) Search items.
        (4) Leave the recommender system.
        """
        call_to_action_template = (
             "{agent_name} is an extremely cautious user who is very particular about what you interact with. {agent_name} will not click on any item unless he/she is absolutely sure he/she likes it. If there is even the slightest hesitation or doubt, {agent_name} will refrain from watching.\n"
            "{agent_name} must choose one of the two actions below:\n"
            "(1) Skip the current video.\n"
            "(2) Watch the video recommended by the recommender system.\n"
            # "(3) Leave the recommender system."
            + "\nTo skip the current recommended video since {agent_name} don't want to waste time on it and prefers to watch other videos that attract him more, write:\n[SKIP] (e.g., [SKIP])."
            + "\nTo watch the current recommended video since it matches {agent_name}'s interests, write:\n[WATCH] (e.g., [WATCH])"
            # + "\n If the video does not match {agent_name}'s interests."
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = full_result.strip()
        # substring = 'must choose one of the four actions below'
        # index = result.find(substring)
        # if index != -1:
        #     # 使用replace方法删除子字符串
        #     result = result[:index-2]
        # print(f'-----------------------------{result}')

        if result.find("[WATCH]") != -1:
            choice = "[WATCH]"
            action = f"{self.name} watches the current video."
            # self.memory.save_context(
            #     {},
            #     {
            #         self.memory.add_memory_key: f"{self.name} took action: {result}",
            #         self.memory.now_key: now,
            #     },
            # )
        else:
            choice = "[SKIP]"
            action = f"{self.name} skip the current video."

        return choice, action

    def generate_feeling(self, observation: str, now) -> str:
        """Feel about each item bought."""
        call_to_action_template = (
            "{agent_name}, how did you feel about the video you just watched? Describe your feelings in one line."
            + "NOTE: Please answer in the first-person perspective."
            + "\n\n"
        )

        full_result = self._generate_reaction(observation, call_to_action_template, now)
        results = full_result.split(".")
        feelings = ""
        for result in results:
            if result.find("language model") != -1:
                break
            feelings += result
        if feelings == "":
            results = full_result.split(",")
            for result in results:
                if result.find("language model") != -1:
                    break
                feelings += result
        # self.memory.save_context(
        #     {},
        #     {
        #         self.memory.add_memory_key: f"{self.name} felt: " f"{feelings}",
        #         self.memory.now_key: now,
        #     },
        # )
        return feelings

    def generate_rec_list_rating(self, observation: str, now) -> str:
        """Generate rating for the recommended movie list."""
        call_to_action_template = "Assuming you are {agent_name}. Very Critically consider each movie in the recommended list and evaluate how well it's genre aligns with your personal interests. For every movie that does not match your interests, the overall rating should be reduced accordingly and obviously. After this thorough assessment, provide an overall rating for the entire list on a scale from 1 to 10, where 1 indicates a poor overall match and 10 indicates an excellent overall match. Your response should be a single integer, reflecting a meticulous evaluation of how each movie fits with your interests. Your response should be a single integer, representing a rigorous and sensitive evaluation of the recommendation quality, with no additional commentary."
        full_result = self._generate_reaction(observation, call_to_action_template, now)
        match = re.search(r'\d+', full_result)
        rating = 0
        if match:
            rating = int(match.group())
        return rating

    def generate_rec_rating(self, observation: str, now) -> str:
        """Generate rating for a movie."""
        call_to_action_template = "Assuming you are {agent_name}. Based on your liking for similar types of movies and your historical ratings, please Very Critically rate this movie on a scale from 1 to 5, where 1 means you really dislike it and 5 means you really like it. Consider that if this movie does not align well with your historical interests, your rating might be lower. Please provide your rating in the following format: 'Rating: [Your Score]'."
        full_result = self._generate_reaction(observation, call_to_action_template, now)
        rating_regex = r"Rating: (\d)"
        extracted_rating = re.search(rating_regex, full_result)
        rating = 0
        if extracted_rating:
            rating = extracted_rating.group(1)
            # print(f"Extracted Rating: {rating}")
        return rating




    # def publish_posting(self, observation, now) -> str:
    #     """Publish posting to all acquaintances."""
    #     # call_to_action_template = (
    #     #     "Posts should be related to recent watched movies on recommender systems."
    #     #     "{agent_name} should not say anything about movies that have not watched or heard about."
    #     #     + "\nIf you were {agent_name}, what will you post? Respond in one line."
    #     #     + "\n\n"
    #     # )
    #     call_to_action_template = (
    #         "Posts should be related to {observation} on recommender systems. "
    #         "{agent_name} should not say anything about movies that have not watched or heard about."
    #         + "\nIf you were {agent_name}, what will you post? Respond in one line."
    #         + "\n\n"
    #     )
    #
    #     result = self._generate_reaction(observation, call_to_action_template, now)
    #     self.memory.save_context(
    #         {},
    #         {
    #             self.memory.add_memory_key: f"{self.name} is publishing posting to all acquaintances. {self.name} posted {result}",
    #             self.memory.now_key: now,
    #         },
    #     )
    #     return result

    def update_watched_history(self, items, now=None):
        """Update history by the items bought. If the number of items in the history achieves the BUFFERSIZE, delete the oldest item."""
        self.watched_history.append(items)
        if len(self.watched_history) > self.BUFFERSIZE:
            self.watched_history = self.watched_history[-self.BUFFERSIZE :]

    def update_heared_history(self, items, now=None):
        """Update history by the items heard. If the number of items in the history achieves the BUFFERSIZE, delete the oldest item."""
        self.heared_history.append(items)
        if len(self.heared_history) > self.BUFFERSIZE:
            self.heared_history = self.heared_history[-self.BUFFERSIZE :]
