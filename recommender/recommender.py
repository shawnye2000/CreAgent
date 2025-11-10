import abc
from .model import *
import importlib
import torch
import pandas as pd
from utils import utils
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import namedtuple
from .early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Sampler, TensorDataset
import random

class UserBatchSampler(Sampler):
    def __init__(self, data_users, user_list):
        self.data_users = data_users
        self.user_list = user_list
        self.user_indices = self._group_by_user()


    def _group_by_user(self):
        # 将每个用户的数据的索引分组
        user_indices = {user: [] for user in self.user_list}
        # print(f'user list:{self.user_list}')
        for idx, user_data in enumerate(self.data_users):
            # print(f'user_data:{user_data}')
            user_id = user_data[0].cpu().item()
            # print(f'user_id:{user_id}')
            user_indices[user_id].append(idx)
        indices = [user_indices[u] for u in self.user_list]
        return indices   #list(user_indices.values())

    def __iter__(self):
        # 按照用户分批返回索引
        for indices in self.user_indices:
            yield indices

    def __len__(self):
        # 返回批次数量（不同的用户数）
        return len(self.user_indices)

class Recommender:
    """
    Recommender System class
    """

    def __init__(self, config, logger, data):
        self.data = data
        self.config = config
        self.logger = logger
        self.random_k = config["rec_random_k"]
        module = importlib.import_module("recommender.model")

        input_data = self.load_dataset_and_post_process()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if self.config['rec_model'] == 'DIN':
            self.model = getattr(module, config["rec_model"])(
                config, input_data, self.device
            )
        else:
            self.model = getattr(module, config["rec_model"])(
                config
            )
        if self.config['rec_model'] != 'Random':
            self.model = self.model.to(self.device)

        if config["reranking_model"]:
            reranking_module = importlib.import_module("recommender.reranking_model")
            self.reranking_model = getattr(reranking_module, config["reranking_model"])(
                self.config['TopK'], self.config['tradeoff_para']
            )
        self.criterion = nn.BCEWithLogitsLoss()
        if self.config['rec_model'] != 'Random':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.epoch_num = config["epoch_num"]
        self.train_data = []

        self.record = {}
        self.round_record = []
        self.positive = {}
        self.inter_df = None
        self.inter_num = 0
        for user in self.data.get_full_users():
            self.record[user] = []
            self.positive[user] = []
            # self.round_record[user] = []


    def train_BPR(self):
        if len(self.train_data) == 0:
            return
        users = []
        items = []
        neg_items = []
        for x in self.train_data:
            if x[7] == 1:
                user = x[0]
                pos_item = x[1]
                all_neg_items = [x[1] for x in self.train_data if x[0] == user]
                if len(all_neg_items) < 10:
                    sample_neg_items = all_neg_items
                else:
                    sample_neg_items = random.sample(all_neg_items, k=10)
                for n_item in sample_neg_items:
                    users.append(user)
                    items.append(pos_item)
                    neg_items.append(n_item)

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(users).to(self.device),
            torch.LongTensor(items).to(self.device),
            torch.LongTensor(neg_items).to(self.device)
        )

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        self.model.train()

        # 假设model、optimizer、criterion等已初始化
        early_stopping = EarlyStopping(patience=5, verbose=True)

        for epoch in tqdm(range(self.epoch_num)):
            epoch_loss = 0.0
            for sample in train_loader:
                sample = sample
                user, item, neg_item = sample
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss(user, item, neg_item)

                # print(f"epoch:{epoch}\n loss:{loss}\n")

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.logger.info(
                f"Epoch {epoch + 1}/{self.epoch_num}, Loss: {epoch_loss / len(train_loader)}"
            )

            early_stopping(epoch_loss / len(train_loader), self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    def train(self):
        if len(self.train_data) == 0:
            return

        users = [x[0] for x in self.train_data]
        items = [x[1] for x in self.train_data]
        channels = [x[2] for x in self.train_data]
        cates = [x[3] for x in self.train_data]
        history_items = [x[4] for x in self.train_data]
        history_channels = [x[5] for x in self.train_data]
        history_cates = [x[6] for x in self.train_data]
        labels = [x[7] for x in self.train_data]
        # tensor_dataset = dataset.TensorDataset(
        #     torch.LongTensor(uid),
        #     torch.LongTensor(item_id), torch.LongTensor(item_brand_id), torch.LongTensor(item_cate_id),
        #     torch.LongTensor(history_item_id), torch.LongTensor(history_brand_id), torch.LongTensor(history_cate_id),
        #     torch.tensor(label, dtype=torch.float32)
        # )
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(users).to(self.device),
            torch.LongTensor(items).to(self.device),
            torch.LongTensor(channels).to(self.device),
            torch.LongTensor(cates).to(self.device),
            torch.LongTensor(history_items).to(self.device),
            torch.LongTensor(history_channels).to(self.device),
            torch.LongTensor(history_cates).to(self.device),
            torch.LongTensor(labels).to(self.device)
        )

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        self.model.train()


        # 假设model、optimizer、criterion等已初始化
        early_stopping = EarlyStopping(patience=5, verbose=True)

        for epoch in tqdm(range(self.epoch_num)):
            epoch_loss = 0.0
            for sample in train_loader:
                sample = sample
                user, item, channel, cate, history_item, history_channel, history_cate, label = sample
                feat = [user, item, channel, cate, history_item, history_channel, history_cate]
                self.optimizer.zero_grad()
                if self.config['rec_model'] == 'DIN':
                    outputs = self.model(feat)
                    loss = self.criterion(outputs, label.float())
                elif self.config['rec_model'] == 'Pop':
                    loss = self.model.calculate_loss(item)
                else:
                    outputs = self.model(user, item)
                    loss = self.criterion(outputs, label.float())

                # print(f"epoch:{epoch}\n loss:{loss}\n")

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()



            self.logger.info(
                f"Epoch {epoch+1}/{self.epoch_num}, Loss: {epoch_loss/len(train_loader)}"
            )

            early_stopping(epoch_loss/len(train_loader), self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))



    def get_full_sort_items_for_users(self, user_list,round_cnt, item_set=[]):
        if item_set == []:
            items = self.data.get_full_items()
        else:
            items = item_set

        data_users = []
        data_items = []
        data_channels = []
        data_cates = []
        data_history_items = []
        data_history_channels = []
        data_history_cates = []
        data_labels = []
        for user in user_list:
            items_tensor = torch.tensor(items).to(self.device)
            all_users = [user for i in range(len(items))]
            all_items = items.copy()
            all_channels = [self.data.get_provider_id_by_item(i) for i in items]
            all_cates = [self.data.get_cate_id_by_item(i) for i in items]
            history_items = self.positive[user]
            if len(history_items) < self.config['max_seq_len']:
                history_items = history_items + [0] * (self.config['max_seq_len'] - len(history_items))
            else:
                history_items = history_items[- self.config['max_seq_len']:]
            history_channels = [self.data.get_provider_id_by_item(i) for i in history_items]
            history_cates = [self.data.get_cate_id_by_item(i) for i in history_items]

            all_his_items = [history_items for i in range(len(items))]
            all_his_chas = [history_channels for i in range(len(items))]
            all_his_cates = [history_cates for i in range(len(items))]

            data_users = data_users + all_users
            data_items = data_items + all_items
            data_channels = data_channels + all_channels
            data_cates = data_cates + all_cates
            data_history_items = data_history_items + all_his_items
            data_history_channels = data_history_channels + all_his_chas
            data_history_cates = data_history_cates + all_his_cates

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(data_users).to(self.device), torch.LongTensor(data_items).to(self.device),
            torch.LongTensor(data_channels).to(self.device), torch.LongTensor(data_cates).to(self.device),
            torch.LongTensor(data_history_items).to(self.device), torch.LongTensor(data_history_channels).to(self.device),
            torch.LongTensor(data_history_cates).to(self.device)
        )

        user_batch_sampler = UserBatchSampler(dataset, user_list)
        infer_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=user_batch_sampler #batch_size=len(items), shuffle=False
        )

        users_sorted_items = {}
        users_item_decs = {}
        # print(f'len user:{len(user_list)}')
        # print(f'len loader:{len(infer_loader)}')
        if len(user_list) != len(infer_loader):
            raise ValueError('User loader size not match')

        UI_matrix = []
        no_reranking_list = []
        for user, samp in zip(user_list, infer_loader):  # user_id and batch is one on one
            user_f, item, channel, cate, history_item, history_channel, history_cate = samp
            feat = [user_f, item, channel, cate, history_item, history_channel, history_cate]
            sorted_items, predicted_ratings = self.model.get_full_sort_items(feat, items_tensor)

            # predicted_ratings is the relevance score for (user, items) for item in items
            #
            no_reranking_list.append([item for item in sorted_items if item not in self.record[user]][:self.config['TopK']])
            for item_id in self.record[user]:
                if item_id in items:
                    iidx = items.index(item_id)  # item id 在items中的index
                    predicted_ratings[iidx] = float('-inf')  #-999
                else:
                    continue
            UI_matrix.append(predicted_ratings)

            # print(f'predict ratings:{predicted_ratings}')
        print(f'round_cnt:{round_cnt} reranking step:{self.config["reranking_start_step"]}')
        if round_cnt <= self.config['reranking_start_step'] or (not self.config['reranking_model']):
            print(f'NO RERANKING round_cnt:{round_cnt}, reranking step:{self.config["reranking_start_step"]} ')
            recommend_list = no_reranking_list
        else:
            print(f'begin reranking...')
            UI_matrix = np.array(UI_matrix)
            if self.config['reranking_model'] in ['MMR', 'DPP', 'APDR']:  # diversity
                M = self.data.get_item_genre_matrix(items)
            else:  # fairness
                M = self.data.get_item_provider_matrix(items)
            rho = np.sum(M, axis=0) / M.shape[0]  # T*K*
            recommend_indices = self.reranking_model.recommendation(UI_matrix, M, rho)
            recommend_list = []
            for rec_indice in recommend_indices:
                recommend_list.append([items[i] for i in rec_indice])


        for user, user_recommend_list in zip(user_list, recommend_list):
            sorted_items = [item for item in user_recommend_list if item not in self.record[user]]
            sorted_item_names = self.data.get_item_names(sorted_items)
            sorted_item_tags = self.data.get_item_tags(sorted_items)
            description = self.data.get_item_description_by_id(sorted_items)
            items = [
                sorted_item_names[i]
                + ";; Genre: "
                + self.data.get_genres_by_ids([sorted_items[i]])[0]
                + ";; Tags: "
                + sorted_item_tags[i]
                + ";; Description: "
                + description[i]
                for i in range(len(sorted_item_names))
            ]
            users_sorted_items[user] = sorted_items  # for user index in users[index]
            users_item_decs[user] = items

        return users_sorted_items, users_item_decs


    def get_full_sort_items_for_MF(self, user_list, round_cnt, item_set=[]):
        """
        Get a list of sorted items for a given user.
        """
        users_sorted_items = {}
        users_item_decs = {}
        self.logger.info(f'{self.config["rec_model"]}')
        if item_set == []:
            items = self.data.get_full_items()
        else:
            items = item_set
        UI_matrix = []
        no_reranking_list = []
        for user in user_list:
            user_tensor = torch.tensor(user).to(self.device)
            items_tensor = torch.tensor(items).to(self.device)
            sorted_items, predicted_ratings = self.model.get_full_sort_items(user_tensor, items_tensor)

            # predicted_ratings is the relevance score for (user, items) for item in items
            #
            no_reranking_list.append([item for item in sorted_items if item not in self.record[user]][:self.config['TopK']])
            for item_id in self.record[user]:
                if item_id in items:
                    iidx = items.index(item_id)  # item id 在items中的index
                    predicted_ratings[iidx] = float('-inf')  #-999
                else:
                    continue
            UI_matrix.append(predicted_ratings)

            # print(f'predict ratings:{predicted_ratings}')
        print(f'round_cnt:{round_cnt} reranking step:{self.config["reranking_start_step"]}')
        if round_cnt <= self.config['reranking_start_step'] or (not self.config['reranking_model']):
            print(f'NO RERANKING round_cnt:{round_cnt}, reranking step:{self.config["reranking_start_step"]} ')
            recommend_list = no_reranking_list
        else:
            print(f'begin reranking...')
            UI_matrix = np.array(UI_matrix)
            if self.config['reranking_model'] in ['MMR', 'DPP', 'APDR']:  # diversity
                M = self.data.get_item_genre_matrix(items)
            else:  # fairness
                M = self.data.get_item_provider_matrix(items)
            rho = np.sum(M, axis=0) / M.shape[0]  # T*K*
            recommend_indices = self.reranking_model.recommendation(UI_matrix, M, rho)
            recommend_list = []
            for rec_indice in recommend_indices:
                recommend_list.append([items[i] for i in rec_indice])




        for user, user_recommend_list in zip(user_list, recommend_list):
            sorted_items = [item for item in user_recommend_list if item not in self.record[user]]
            sorted_item_names = self.data.get_item_names(sorted_items)
            sorted_item_tags = self.data.get_item_tags(sorted_items)
            description = self.data.get_item_description_by_id(sorted_items)
            items = [
                sorted_item_names[i]
                + ";; Genre: "
                + self.data.get_genres_by_ids([sorted_items[i]])[0]
                + ";; Tags: "
                + sorted_item_tags[i]
                + ";; Description: "
                + description[i]
                for i in range(len(sorted_item_names))
            ]
            users_sorted_items[user] = sorted_items  # for user index in users[index]
            users_item_decs[user] = items
        return users_sorted_items, users_item_decs
    def get_full_sort_items(self, user, item_set=[]):
        """
        Get a list of sorted items for a given user.
        """
        # self.logger.info(f'{self.config["rec_model"]}')

        if self.config['rec_model'] == "MF":
            items = self.data.get_full_items()
            user_tensor = torch.tensor(user).to(self.device)
            items_tensor = torch.tensor(items).to(self.device)
            sorted_items = self.model.get_full_sort_items(user_tensor, items_tensor)
        else:
            if item_set == []:
                items = self.data.get_full_items()
            else:
                items = item_set
            # self.logger.info(f'{items}')
            items_tensor = torch.tensor(items).to(self.device)
            all_users = [user for i in range(len(items))]
            all_items = items.copy()
            all_channels = [self.data.get_provider_id_by_item(i) for i in items]
            all_cates = [self.data.get_cate_id_by_item(i) for i in items]
            history_items = self.positive[user]
            if len(history_items) < self.config['max_seq_len']:
                history_items = history_items + [0] * (self.config['max_seq_len'] - len(history_items))
            else:
                history_items = history_items[- self.config['max_seq_len']:]
            history_channels = [self.data.get_provider_id_by_item(i) for i in history_items] #* len(items)
            history_cates = [self.data.get_cate_id_by_item(i) for i in history_items] #* len(items)

            all_his_items = [history_items for i in range(len(items))]
            all_his_chas = [history_channels for i in range(len(items))]
            all_his_cates = [history_cates for i in range(len(items))]

            dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(all_users).to(self.device), torch.LongTensor(all_items).to(self.device),
                torch.LongTensor(all_channels).to(self.device), torch.LongTensor(all_cates).to(self.device),
                torch.LongTensor(all_his_items).to(self.device), torch.LongTensor(all_his_chas).to(self.device),
                torch.LongTensor(all_his_cates).to(self.device)
            )

            infer_loader = torch.utils.data.DataLoader(
                dataset, batch_size=len(items)
            )
            for samp in infer_loader:
                user_f, item, channel, cate, history_item, history_channel, history_cate = samp
                feat = [user_f, item, channel, cate, history_item, history_channel, history_cate]
                sorted_items, sorted_ratings = self.model.get_full_sort_items(feat, items_tensor)




        # reranked_items = self.reranking_model.get_reranked_items(sorted_items, sorted_ratings)

        sorted_items = [item for item in sorted_items if item not in self.record[user]]
        sorted_item_names = self.data.get_item_names(sorted_items)
        sorted_item_tags = self.data.get_item_tags(sorted_items)
        description = self.data.get_item_description_by_id(sorted_items)
        items = [
            sorted_item_names[i]
            + ";; Genre: "
            + self.data.get_genres_by_ids([sorted_items[i]])[0]
            + ";; Tags: "
            + sorted_item_tags[i]
            + ";; Description: "
            + description[i]
            for i in range(len(sorted_item_names))
        ]
        return sorted_items, items

    def get_random_items(self, user, items):
        """
        Get a list of sorted items for a given user.
        """
        import random
        items = self.data.get_full_items()
        sorted_items = random.sample(items, k=8)
        random.shuffle(sorted_items)
        # user_tensor = torch.tensor(user)
        # items_tensor = torch.tensor(items)
        # sorted_items = self.model.get_full_sort_items(user_tensor, items_tensor)
        # if self.random_k > 0 and random == True:
        #     sorted_items = self.add_random_items(user, sorted_items)
        # if random == True:
        # sorted_items = items.copy()
        #
        # random.shuffle(sorted_items)
        # print(f'sorted_items:{sorted_items}')

        # sorted_items = [item for item in sorted_items if item not in self.record[user]]

        sorted_item_names = self.data.get_item_names(sorted_items)
        description = self.data.get_item_description_by_id(sorted_items)
        items = [
            sorted_item_names[i]
            + ";; Genre: "
            + self.data.get_genres_by_ids([sorted_items[i]])
            + ";;"
            + description[i]
            for i in range(len(sorted_item_names))
        ]
        return sorted_items, items

    def get_item(self, idx):
        item_name = self.data.get_item_names([idx])[0]
        description = self.data.get_item_description_by_id([idx])[0]
        item = item_name + ";;" + description
        return item

    def get_search_items(self, item_name):
        return self.data.search_items(item_name)

    def get_inter_num(self):
        return self.inter_num

    def update_history_by_name(self, user_id, item_names):
        """
        Update the history of a given user.
        """
        item_names = [item_name.strip(" <>'\"") for item_name in item_names]
        item_ids = self.data.get_item_ids(item_names)
        self.record[user_id].extend(item_ids)

    def update_history_by_id(self, user_id, item_ids):
        """
        Update the history of a given user.
        """
        self.record[user_id].extend(item_ids)

    def update_positive(self, user_id, item_names):
        """
        Update the positive history of a given user.
        """
        item_ids = self.data.get_item_ids(item_names)
        if len(item_ids) == 0:
            return
        self.positive[user_id].extend(item_ids)
        self.inter_num += len(item_ids)

    def update_positive_by_id(self, user_id, item_id):
        """
        Update the history of a given user.
        """
        self.positive[user_id].append(item_id)


    def save_round_interaction(self, round_cnt):
        df = pd.DataFrame(self.round_record)
        df.to_csv(
            self.config["interaction_path"],
            index=False,
            mode='a' if round_cnt != 1 else 'w',
            header=False if round_cnt != 1 else True
        )


    def save_interaction(self, round_cnt):
        """
        Save the interaction history to a csv file.
        """

        inters = []
        users = self.data.get_full_users()
        for user in users:
            for item in self.positive[user]:
                provider_name = self.data.items[item]["provider_name"]
                genre = self.data.items[item]["genre"]
                new_row = {"round_cnt": round_cnt ,"user_id": user, "item_id": item, "provider_name": provider_name, "genre": genre, "rating": 1}
                inters.append(new_row)

            for item in self.record[user]:
                provider_name = self.data.items[item]["provider_name"]
                genre = self.data.items[item]["genre"]
                if item in self.positive[user]:
                    continue
                new_row = {"round_cnt": round_cnt ,"user_id": user, "item_id": item, "provider_name": provider_name, "genre": genre, "rating": 0}
                inters.append(new_row)

        df = pd.DataFrame(inters)
        df.to_csv(
            self.config["interaction_path"],
            index=False,
        )
        self.inter_df = df



    # tensor_dataset = dataset.TensorDataset(
    #     torch.LongTensor(uid),
    #     torch.LongTensor(item_id), torch.LongTensor(item_brand_id), torch.LongTensor(item_cate_id),
    #     torch.LongTensor(history_item_id), torch.LongTensor(history_brand_id), torch.LongTensor(history_cate_id),
    #     torch.tensor(label, dtype=torch.float32)
    # )
    def add_train_data(self, user, item, label):
        item_channel_id = self.data.get_provider_id_by_item(item)
        item_cate_id = self.data.get_cate_id_by_item(item)
        history_item_ids = self.positive[user]

        if len(history_item_ids) < self.config['max_seq_len']:
            history_item_ids = history_item_ids + [0]* (self.config['max_seq_len']- len(history_item_ids))
        else:
            history_item_ids = history_item_ids[- self.config['max_seq_len']:]
        history_cate_ids = [self.data.get_cate_id_by_item(i) for i in history_item_ids]
        history_channel_ids = [self.data.get_provider_id_by_item(i) for i in history_item_ids]

        self.train_data.append((user, item, item_channel_id, item_cate_id, history_item_ids, history_channel_ids, history_cate_ids, label))


    def clear_train_data(self):
        self.train_data = []

    def add_round_record(self,user, item, label, round_cnt):
        provider_name = self.data.items[item]["provider_name"]
        genre = self.data.items[item]["genre"]
        user_name = self.data.users[user]['name']
        self.round_record.append({"round_cnt":round_cnt, "user_id": user, "user_name": user_name, "item_id": item, "provider_name": provider_name, "genre": genre, "rating": label})

    def get_entropy(
        self,
    ):
        tot_entropy = 0
        for user in self.record.keys():
            inters = self.record[user]
            genres = self.data.get_genres_by_id(inters)
            entropy = utils.calculate_entropy(genres)
            tot_entropy += entropy

        return tot_entropy / len(self.record.keys())


    def update_data(self, data):
        self.data = data

    def load_dataset_and_post_process(self):
        SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
        VarLenSparseFeat = namedtuple('VarLenSparseFeat',
                                      ['name', 'sparsefeat', 'maxlen'])
        # DenseFeat = namedtuple('DenseFeat', ['name', 'embedding_dim'])
        input_data = [
            SparseFeat(name='user_id', vocabulary_size=self.data.get_user_num() + 1,
                       embedding_dim=self.config['user_id_dim']
                       ),
            SparseFeat(name='item_id', vocabulary_size=self.config['max_item_num'] + 1,
                       embedding_dim=self.config['item_id_dim']
                       ),
            SparseFeat(name='item_brand_id', vocabulary_size=self.data.get_provider_num() + 1,
                       embedding_dim=self.config['item_brand_id_dim']
                       ),
            SparseFeat(name='item_cate_id', vocabulary_size=self.data.get_cates_num() + 1,
                       embedding_dim=self.config['item_cate_id_dim']
                       ),
            # DenseFeat(name='item_text_id', embedding_dim=768),
            VarLenSparseFeat(sparsefeat='item_id', maxlen=self.config['max_len'],
                             name='history_item_id'),
            VarLenSparseFeat(sparsefeat='item_brand_id', maxlen=self.config['max_len'],
                             name='history_item_brand_id'),
            VarLenSparseFeat(sparsefeat='item_cate_id', maxlen=self.config['max_len'],
                             name='history_item_cate_id'),
        ]
        return input_data