#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

 CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/train_lora/llama3_lora_ppo.yaml  > /home/xiaopeng_ye/experiment/Agent4Fairness/LLaMA-Factory/log/07101003.log 2>&1 & #examples/extras/fsdp_qlora/llama3_lora_sft.yaml



#nohup FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_ppo.yaml > /home/xiaopeng_ye/experiment/Agent4Fairness/LLaMA-Factory/log/07091003.log 2>&1 &