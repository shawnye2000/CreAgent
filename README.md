# CreAgent
This contains the codes of the simulation platform  equipped with CreAgent, which is used for long-term recommender system evaluation.


First, collected the dataset and put them into `LLaMA-Factory/data` by creating a new folder, e.g., `LLaMA-Factory/data/youtube`. The dataset should contain both creator and user.

The dataset format of provider is 
```
[
    {
        "channel_name": "xxx",
        "history_categories": {
            "Entertainment": 98,
            "Film & Animation": 1,
            "Education": 1
        },
        "creation_frequency": xx,
        "history_items": []
      },
     ...
]
```


The dataset format of user is 
```
[
    {
        "user_name": "xxx",
        "interest": [
            "People & Blogs",
            "Entertainment",
            "Film & Animation"
        ],
        "history": [
            "SS1ac8mAhhE",
            "KB8-cAU8kW4",
            ...
        ]
        },
        ...
]
```

Second, please determine the setups of the simulation platform and modify the config file:

`
cd LLaMA-Factory/src/llamafactory/config/config.yaml
`

To run the simulator, please enter the llama-factory dictionary

`
cd LLaMA-Factory/src/llamafactory/simulator
`

Then, you can run the simulator

`
python simulator.py
`


To change the configure setting, you can enter the `LLaMA-Factory/src/llamafactory/config/config.yaml` file and edit. 



