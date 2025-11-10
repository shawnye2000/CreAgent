from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from openai import OpenAI

class CustomLLM(LLM):
    max_token: int
    URL: str = "http://localhost:8000/v1"
    OPENAI_API_KEY: str = 'EMPTY'
    headers: dict = {"Content-Type": "application/json"}
    payload: dict = {"prompt": "", "history": []}
    logger: Any
    flag: bool = True
    config: dict = None
    def __init__(self, max_token, logger, config):
        super().__init__(
            max_token = max_token,
            logger =logger,
        )
        self.max_token = max_token
        self.logger = logger
        self.config = config


    @property
    def _llm_type(self) -> str:
        return "CustomLLM"

    def _call(
        self,
        prompt: str,
        history = None,
        profile = '',
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if len(profile) > 0:
            messages = [
                {"role": "system", "content": profile},
            ]
        else:
            messages = []
        if history != None:
            for qr_pair in history:
                messages.append(
                    {"role": "user", "content": qr_pair[0]},
                )
                messages.append(
                    {"role": "assistant", "content": qr_pair[1]},
                )

        messages.append(
            {"role": "user", "content": prompt},
        )

        client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['api_base'],
        )

        chat_response = client.chat.completions.create(
            model=self.config['llm_model_name'],
            messages=messages
        )
        chat_response = chat_response.choices[0].message.content

        return chat_response


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"URL": self.URL, "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "max_token": self.max_token}
