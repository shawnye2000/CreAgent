# CreAgent and Simulation platform 

This repo contains the codes of the simulation platform equipped with CreAgent, which is used for long-term recommender system evaluation.
proposed by the SIGIR 2025 paper "**LLM-Empowered Creator Simulation for Long-Term Evaluation of Recommender Systems Under Information Asymmetry**"


# Environment 
Clone the github repo and 
```
git clone https://github.com/shawnye2000/CreAgent.git
cd CreAgent
conda create -n creagent python=3.10
pip install -r requirements.txt
```

Then,  download the dataset from [google drive](https://drive.google.com/drive/folders/1PwNygSNd-L161x-wDmiwq78E0_VwzoMh?usp=sharing). Download the `users.json` and `provider.json` from `Small_YouTube` and put them into `dataset/youtube`. 


Third, please determine the setups of the simulation platform and modify the config file:`config/config.yaml`
```
api_base: http://localhost:8000/v1
api_key: EMPTY
llm_model_name: your_llm_name 
embedding_model_path: your_embedding_model
```
# Running
Activate your conda environment
```
conda activate creagent
```


Load your vllm
```
python -m vllm.entrypoints.openai.api_server    --model your_llm_name   --trust-remote-code    --tensor-parallel-size 2    --api-key EMPTY    --port 8000  --enforce-eager --gpu-memory-utilization 0.9
```

Then, you can run the simulator environment
```
python simulator/simulator.py
```


To change the configure setting, you can enter the `config/config.yaml` file and edit. 



