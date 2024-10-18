import csv
from langchain.vectorstores import FAISS
from ...utils import utils
import pandas as pd
import json
import numpy as np
import random


def json_reader(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        json_str = f.read()
    creator_video_dict = json.loads(json_str)  # every cate provider
    return creator_video_dict

class Data:
    """
    Data class for loading data from local files.
    """

    def __init__(self, config):
        self.config = config
        self.items = {}
        self.users = {}
        self.providers = {}
        self.item2provider = {}
        self.item_token2id = {}
        self.item_provider_matrix = None

        self.db = None
        # self.load_items(config["item_path"])
        self.load_providers(config["provider_path"])
        self.load_users(config["user_path"])
        # self.load_relationship(config["relationship_path"])
        # self.load_faiss_db(config["index_name"])
        # self.item2provider = self.load_item_provider_relationship()

        self.cate2id = self.load_cate2id()

    def load_cate2id(self):
        cates_list = self.get_item_categories()
        cates_num = len(cates_list)

        cates_id_list = [i for i in range(1, cates_num+1)]
        mapped_dict = {key: value for key, value in zip(cates_list, cates_id_list)}
        return mapped_dict

    # def load_item_provider_relationship(self):
    #     provider_name2id =  {}
    #     for id, dic in self.providers.items():
    #         provider_name2id[dic['name']] = id
    #
    #     items_to_remove = []
    #     for item_id in self.items.keys():
    #         provider_name = self.items[item_id]['provider_name']
    #         if provider_name not in provider_name2id.keys():
    #             items_to_remove.append(item_id)
    #
    #
    #     for i in items_to_remove:
    #         del self.items[i]
    #     self.items = {i: value for i, (key, value) in enumerate(self.items.items(), start=1)}
    #     for item_id in self.items.keys():
    #         provider_name = self.items[item_id]['provider_name']
    #         provider_id = provider_name2id[provider_name]
    #         self.items[item_id]['provider_id'] = provider_id
    #         self.providers[provider_id]['items'].append(item_id)
    #
    #
    #     item2provider = {}
    #     for pro in self.providers.keys():
    #         items = self.providers[pro]['items']
    #         for item in items:
    #             item2provider[item] = pro
    #     return item2provider

    def load_faiss_db(self, index_name):
        """
        Load faiss db from local if exists, otherwise create a new one.

        """
        _, embeddings = utils.get_embedding_model()
        try:
            self.db = FAISS.load_local(index_name, embeddings)
            print("Load faiss db from local")
        except:
            titles = [item["name"] for item in self.items.values()]
            self.db = FAISS.from_texts(titles, embeddings)
            self.db.save_local(index_name)


    # def load_items(self, file_path):
    #     """
    #     Load items from local file.
    #     """
    #     with open(file_path, "r", newline="") as file:
    #         reader = csv.reader(file)
    #         next(reader)  # Skip the header line
    #         for row in reader:
    #             # ['video_id', 'title', 'channel_title',	'category_title', 'publish_time',	'tags',	'description']
    #             item_id, title, provider_name, genre, upload_time, tags, description = row
    #             tags = tags.split('|')
    #             tags = [item.replace('""', '').strip() for item in tags]
    #             self.items[int(item_id)] = {
    #                 "name": title.strip(),
    #                 "provider_name": provider_name,
    #                 "genre": genre,
    #                 "upload_time": upload_time,
    #                 "tags": tags,
    #                 "description": description.strip(),
    #                 "inter_cnt": 0,
    #                 "mention_cnt": 0,
    #             }

    def load_users(self, file_path):
        """
        Load users from local file.
        """
        user_json = json_reader(file_path)
        random_users = random.sample(user_json, self.config["agent_num"])  #agent_num
        cnt = 1
        for user_dict in random_users:
            self.users[cnt] = {
                "name": user_dict['user_name'],
                "interest": '|'.join(user_dict['interest']),
                "history": [self.item_token2id[item_token] for item_token in user_dict['history'] if item_token in self.item_token2id.keys()]
            }
            cnt += 1
        # cnt = 1
        # with open(file_path, "r", newline="") as file:
        #     reader = csv.reader(file)
        #     next(reader)  # Skip the header line
        #     for row in reader:
        #         user_id, name, gender, age, traits, status, interest, feature = row
        #         self.users[cnt] = {
        #             "name": name,
        #             "gender": gender,
        #             "age": int(age),
        #             "traits": traits,
        #             "status": status,
        #             "interest": interest,
        #             "feature": feature,
        #         }
        #         cnt += 1
        #         # if self.get_user_num() == self.config["agent_num"]:
        #         #     break
        # selected_users = random.sample(list(self.users.keys()), self.config["agent_num"])
        # self.users = {key: self.users[key] for key in selected_users}
        # self.users = {i: value for i, (key, value) in enumerate(self.users.items(), start=1)}
        print('user interest')
        for u in self.users.values():
            print(f'name:{u["name"]}'
                  f'interest:{u["interest"]}')


    def load_providers(self, file_path):
        """
        Load users from local file.
        """
        # cnt = 0
        providers_json = json_reader(file_path)
        random_providers = random.sample(providers_json, k=self.config["provider_agent_num"])
        cnt = 1
        item_cnt = 1
        max_freq = max([p_d['creation_frequency'] for p_d in random_providers])
        for pro_dict in random_providers:
            self.providers[cnt] = {
                "name": pro_dict["channel_name"],
                "category_history":  pro_dict["history_categories"],
                "items": [],
                "frequency": float(pro_dict['creation_frequency'])  # #/max_freq
            }
            print(f'frequency:{pro_dict["creation_frequency"]}')
            history_items = pro_dict["history_items"]
            for item_dict in history_items:
                self.items[item_cnt] = {
                    "name": item_dict["title"],
                    "token": item_dict['id'],
                    "provider_name": pro_dict["channel_name"],
                    "provider_id": cnt,
                    "tags": [],
                    "genre": item_dict['genre'],
                    "upload_time": -999,
                    "description": item_dict['description'].strip(),
                }
                self.item2provider[item_cnt] = cnt
                self.providers[cnt]['items'].append(item_cnt)
                self.item_token2id[item_dict['id']] = item_cnt
                item_cnt += 1


            cnt += 1

    def get_full_items(self):
        return list(self.items.keys())

    def get_inter_popular_items(self):
        """
        Get the most popular items based on the number of interactions.
        """
        ids = sorted(
            self.items.keys(), key=lambda x: self.items[x]["inter_cnt"], reverse=True
        )[:3]
        return self.get_item_names(ids)

    def add_inter_cnt(self, item_names):
        item_ids = self.get_item_ids(item_names)
        for item_id in item_ids:
            self.items[item_id]["inter_cnt"] += 1

    def add_mention_cnt(self, item_names):
        item_ids = self.get_item_ids(item_names)
        for item_id in item_ids:
            self.items[item_id]["mention_cnt"] += 1

    def get_mention_popular_items(self):
        """
        Get the most popular items based on the number of mentions.
        """
        ids = sorted(
            self.items.keys(), key=lambda x: self.items[x]["mention_cnt"], reverse=True
        )[:3]
        return self.get_item_names(ids)

    def get_item_names(self, item_ids):
        return ["<" + self.items[item_id]["name"] + ">" for item_id in item_ids]

    def get_item_tags(self, item_ids):
        return [f"{self.items[item_id]['tags']}" for item_id in item_ids]

    def get_item_ids(self, item_names):
        item_ids = []
        for item in item_names:
            for item_id, item_info in self.items.items():
                if item_info["name"] in item:
                    item_ids.append(item_id)
                    break
        return item_ids


    def get_all_item_ids(self):
        if self.get_item_num() == 0:
            return []
        else:
            return list(self.items.keys())

    def get_item_categories(self):
        # cate_set = set()
        # for item_id, item_info in self.items.items():
        #     cate_set.add(item_info['genre'])
        #
        # return list(cate_set)
        # return ['People & Blogs', 'Entertainment', 'Film & Animation', 'News & Politics', 'Science & Technology', 'Sports',

        #  'Music', 'Howto & Style', 'Pets & Animals', 'Education', 'Autos & Vehicles']
        genre_list = [
            'Film & Animation',
            'Autos & Vehicles',
            'Music',
            'Pets & Animals',
            'Sports',
            'Travel & Events',
            'Gaming',
            'People & Blogs',
            'Comedy',
            'Entertainment',
            'News & Politics',
            'Howto & Style',
            'Education',
            'Science & Technology',
            'Nonprofits & Activism']
        return genre_list
    def get_item_ids_exact(self, item_names):
        item_ids = []
        for item in item_names:
            for item_id, item_info in self.items.items():
                if item_info["name"] == item:
                    item_ids.append(item_id)
                    break
        return item_ids

    def get_full_users(self):
        return list(self.users.keys())

    def get_user_names(self, user_ids):
        return [self.users[user_id]["name"] for user_id in user_ids]

    def get_user_ids(self, user_names):
        user_ids = []
        for user in user_names:
            for user_id, user_info in self.users.items():
                if user_info["name"] == user:
                    user_ids.append(user_id)
                    break
        return user_ids

    def get_user_num(self):
        """
        Return the number of users.
        """
        return len(self.users.keys())

    def get_provider_num(self):
        """
        Return the number of providers.
        """
        return len(self.providers.keys())

    def get_item_num(self):
        """
        Return the number of items.
        """
        return len(self.items.keys())

    def get_cates_num(self):
        return len(self.get_item_categories())  #len(self.items.keys())




    def search_items(self, item, k=50):
        """
        Search similar items from faiss db.
        Args:
            item: str, item name
            k: int, number of similar items to return
        """
        docs = self.db.similarity_search(item, k)
        item_names = [doc.page_content for doc in docs]
        return item_names

    def get_all_contacts(self, user_id):
        """
        Get all contacts of a user.
        """
        if "contact" not in self.users[user_id]:
            print(f"{user_id} has no contact.")
            return []
        ids = []
        for id in self.users[user_id]["contact"]:
            if id < self.get_user_num():
                ids.append(id)
        return self.get_user_names(ids)

    def get_all_contacts_id(self, user_id):
        """
        Get all contacts of a user.
        """
        if "contact" not in self.users[user_id]:
            print(f"{user_id} has no contact.")
            return []
        ids = []
        for id in self.users[user_id]["contact"]:
            if id < self.get_user_num():
                ids.append(id)
        return ids

    def get_relationships(self, user_id):
        """
        Get all relationships of a user.
        """
        if "contact" not in self.users[user_id]:
            print(f"{user_id} has no contact.")
            return dict()
        return self.users[user_id]["contact"]

    def get_relationship_names(self, user_id):
        """
        Get all relationship IDs of a user.
        """
        if "contact" not in self.users[user_id]:
            print(f"{user_id} has no contact.")
            return dict()
        relatiobnships = dict()
        for id in self.users[user_id]["contact"]:
            relatiobnships[self.users[id]["name"]] = self.users[user_id]["contact"][id]
        return relatiobnships



    def get_item_description_by_id(self, item_ids):
        """
        Get description of items by item id.
        """
        return [self.items[item_id]["description"] for item_id in item_ids]

    def get_item_description_by_name(self, item_names):
        """
        Get description of items by item name.
        """
        item_descriptions = []
        for item in item_names:
            found = False
            for item_id, item_info in self.items.items():
                if item_info["name"] == item.strip(" <>"):
                    item_descriptions.append(item_info["description"])
                    found = True
                    break
            if not found:
                item_descriptions.append("")
        return item_descriptions

    def get_genres_by_ids(self, item_ids):
        """
        Get genre of items by item id.
        """
        return [self.items[item_id]["genre"] for item_id in item_ids]
        # return [
        #     genre
        #     for item_id in item_ids
        #     for genre in self.items[item_id]["genre"].split('|')
        # ]

    def get_cate_id_by_item(self, item_id):
        if item_id == 0:
            return 0
        cate_name = self.items[item_id]["genre"]
        try:
            cate_id = self.cate2id[cate_name]
        except TypeError as e:
            print(e)
            print(f'WrongCateDict:{self.cate2id}\n Cate Name:{cate_name}')
            cate_id = 0
        return cate_id

    def get_provider_id_by_item(self, item_id):
        if item_id == 0:
            return 0
        return self.item2provider[item_id]





      
    def get_provider_id_by_name(self, name):
        for pro_id, pro_dict in self.providers.items():
            if pro_dict['name'] == name:
                return pro_id

        raise ValueError(f'Provider {name} NOT FOUND')

    def get_provider_id_by_item_id(self, item_id):
        return self.item2provider[item_id]

    def get_user_interest_dict(self):
        interest_count = {k: 0 for k in self.get_item_categories()}
        for u_id, u_dic in self.users.items():
            u_interest_str = u_dic['interest']
            if u_interest_str.find('|') != -1:
                u_interest_list = u_interest_str.split('|')
            else:
                u_interest_list = [u_interest_str]

            for u_inter in u_interest_list:
                interest_count[u_inter] += 1
        return interest_count


    def get_item_provider_matrix(self, items):
        provider_list = []
        for item_id in items:
            if self.item2provider[item_id] in provider_list:
                continue
            else:
                provider_list.append(self.item2provider[item_id])
        provider_num = len(provider_list)

        M = np.zeros((len(items), provider_num))

        provider_dict = {p: i for i, p in enumerate(provider_list)}
        # provider_dict = {self.item2provider[item_id]: i for i, item_id in enumerate(items)}
        for i, item_id in enumerate(items):
            pindex = provider_dict[self.item2provider[item_id]]
            M[i, pindex] = 1
        return M

    def get_item_genre_matrix(self, items):
        genre_list = []
        for item_id in items:
            if self.get_cate_id_by_item(item_id) in genre_list:
                continue
            else:
                genre_list.append(self.get_cate_id_by_item(item_id))
        genre_num = len(genre_list)

        M = np.zeros((len(items), genre_num))

        genre_dict = {p: i for i, p in enumerate(genre_list)}
        # provider_dict = {self.item2provider[item_id]: i for i, item_id in enumerate(items)}
        for i, item_id in enumerate(items):
            gindex = genre_dict[self.get_cate_id_by_item(item_id)]
            M[i, gindex] = 1
        return M

