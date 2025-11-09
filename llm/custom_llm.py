from typing import Any, List, Mapping, Optional
from vllm import LLM as vvLLM
from vllm import SamplingParams
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["LOCAL_RANK"] = "1"
class CustomLLM(LLM):
    max_token: int
    URL: str = "http://localhost:8000/v1"
    OPENAI_API_KEY: str = 'EMPTY'
    headers: dict = {"Content-Type": "application/json"}
    payload: dict = {"prompt": "", "history": []}
    logger: Any
    flag: bool = True
    # tokenizer: Any
    # # model: Any
    # llm: Any
    # sampling_params: Any
    def __init__(self, max_token, logger):
        super().__init__(
            max_token = max_token,
            logger =logger,
            # llm = LLM,
            # tokenizer = tokenizer
        )
        self.max_token = max_token
        self.logger = logger
        # self.model = model.module   #self.accelerator.unwrap_model(self.model)
        # self.tokenizer = tokenizer

        # self.llm = torch.nn.DataParallel(self.llm, device_ids=list(range(2)))
        # self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=self.max_token,
                                              # repetition_penalty=1.05,
                                              # stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])


    @property
    def _llm_type(self) -> str:
        return "CustomLLM"

    def _call(
        self,
        prompt: str,
        history = None,
        profile = '',
        stop: Optional[List[str]] = None,
        # history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # prompt_data = json.dumps(prompt, ensure_ascii=False)

        # from accelerate import infer_auto_device_map
        #
        # device_map = infer_auto_device_map(my_model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})

        # model = AutoModelForCausalLM.from_pretrained(
        #     '/home/xiaopeng_ye/LLMs/Meta-Llama-3-8B-Instruct',
        #     torch_dtype=torch.float16,
        #     device_map='auto',#"sequential",
        #     trust_remote_code=True
        # )
        # model.half()
        # tokenizer = AutoTokenizer.from_pretrained(
        #     '/home/xiaopeng_ye/LLMs/Meta-Llama-3-8B-Instruct', trust_remote_code=True)

        # tokenizer.half()
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
        # print(f'messages:{messages}')
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        chat_response = client.chat.completions.create(
            model="llama3-8b",
            messages=messages
        )
        chat_response = chat_response.choices[0].message.content
        # print("Chat response:", chat_response)
        # print(messages)

        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     return_tensors="pt",
        #     add_generation_prompt=True
        # )
        # # print(f'---------text:{text}')
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(
        #     'cuda')  # if args.model != 'glm' else text.to('cuda')
        #
        # #
        # # outputs = self.llm.module.generate(model_inputs, self.sampling_params)
        # generated_ids = self.model.generate(
        #     model_inputs.input_ids,
        #     max_new_tokens=512,
        #     pad_token_id=self.tokenizer.eos_token_id
        # )
        #
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        # #
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        # print(f'---------response:{response}')
        return chat_response


    # def _call(
    #     self,
    #     prompt: str,
    #     history = None,
    #     profile = '',
    #     stop: Optional[List[str]] = None,
    #     # history: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    # ) -> str:
    #
    #     if len(profile) > 0:
    #         messages = [
    #             {"role": "system", "content": profile},
    #         ]
    #     else:
    #         messages = []
    #     if history != None:
    #         for qr_pair in history:
    #             messages.append(
    #                 {"role": "user", "content": qr_pair[0]},
    #             )
    #             messages.append(
    #                 {"role": "assistant", "content": qr_pair[1]},
    #             )
    #
    #     messages.append(
    #         {"role": "user", "content": prompt},
    #     )
    #
    #     prompts = self.tokenizer.apply_chat_template(messages,
    #                                     tokenize=False)
    #                                     # add_generation_prompt = True)
    #     outputs = self.llm.generate(prompts,
    #                                 self.sampling_params)
    #
    #     output = outputs[0].outputs[0].text
    #
    #     return output
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        # return {
        #     "max_token": self.max_token,
        #     "URL": self.URL,
        #     "headers": self.headers,
        #     "payload": self.payload,
        # }

        return {"URL": self.URL, "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "max_token": self.max_token}
