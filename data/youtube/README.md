
Download the providers.json file from the [google drive](https://drive.google.com/drive/folders/1CcG8a7qE-rDnVXJPou3OI7b1-P9xy-B3)


# Big YouTube 

The Big Youtube dataset is a merged dataset with item description, user comments, and provider profile.


# Small YouTube

The Small YouTube dataset is processed and sampled from **Big YouTube **.

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
