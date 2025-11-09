import numpy as np
import heapq
import os
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time

from numpy import mean
from tqdm import tqdm

import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# def construct_list_with_score(data_dir, max_time_len):
#     user, profile, itm_spar, itm_dens, label, pos, list_len, rank_score = pickle.load(open(data_dir, 'rb'))
#     print(len(user), len(itm_spar))
#     cut_itm_dens, cut_itm_spar, cut_label, cut_pos, cut_score, cut_usr_spar, cut_usr_dens, de_label, cut_hist_pos = [], [], [], [], [], [], [], [], []
#     for i, itm_spar_i, itm_dens_i, label_i, pos_i, list_len_i, score_i in zip(list(range(len(label))),
#                                                                      itm_spar, itm_dens, label, pos, list_len, rank_score):

#         if len(itm_spar_i) >= max_time_len:
#             cut_itm_spar.append(itm_spar_i[: max_time_len])
#             cut_itm_dens.append(itm_dens_i[: max_time_len])
#             cut_label.append(label_i[: max_time_len])
#             # de_label.append(de_lb[: max_time_len])
#             cut_pos.append(pos_i[: max_time_len])
#             list_len[i] = max_time_len
#             cut_score.append(score_i[: max_time_len])
#         else:
#             cut_itm_spar.append(
#                 itm_spar_i + [np.zeros_like(np.array(itm_spar_i[0])).tolist()] * (max_time_len - len(itm_spar_i)))
#             cut_itm_dens.append(
#                 itm_dens_i + [np.zeros_like(np.array(itm_dens_i[0])).tolist()] * (max_time_len - len(itm_dens_i)))
#             cut_label.append(label_i + [0 for _ in range(max_time_len - list_len_i)])
#             cut_score.append(score_i + [float('-inf') for _ in range(max_time_len - list_len_i)])
#             # de_label.append(de_lb + [0 for _ in range(max_time_len - list_len_i)])
#             cut_pos.append(pos_i + [j for j in range(list_len_i, max_time_len)])

#     return user, profile, cut_itm_spar, cut_itm_dens, cut_label, cut_pos, list_len, cut_score

class MMR(object):
    def __init__(self, TopK, lbd):
        self.TopK = TopK
        self.lamda = lbd
    """
    	output_dict = self.predict(dataset) ### {"predictions":predictions([2874（all) x 100]),"dataset_ids":dataset_ids([2874（all) x 100])} 
		baseline = MMR()
		output_dict = baseline.post_process(output_dict, id_multihot,lamda=1)
    """

    def softmax(selfd, x: np.array):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def sim_scores(self, cate_id):
        # 矩阵乘法
        sim_matrix = np.array(cate_id) @ np.array(cate_id).T
        sum = np.sum(sim_matrix, axis=1)
        sum = np.sum(sim_matrix, axis=1) - np.diag(sim_matrix)
        # 按照行求和
        sim_scores = sum / np.max(sum) + 1  # self.softmax(np.sum(sim_matrix, axis=1) )
        # yanzhen = np.sum(sim_scores)
        return sim_scores

    def sim(self, current_id, chosen_ids, cate_id: dict):
        simmax = 0
        for i in chosen_ids:
            sim = np.dot(cate_id[current_id], cate_id[i])
            if sim > simmax:
                simmax = sim

        # for i in chosen_ids:
        #     sim = np.dot(cate_id[current_id], cate_id[i])
        #     if sim > 0 :
        #         return float('inf')
        # 随机0-0.5之间的数
        return simmax

    def recommendation(self, UI_matrix, M, rho):

        # id_multihot =
        lamda = self.lamda
        rank_scores = UI_matrix  # [batch_size, 100]
        item_num = UI_matrix.shape[1]

        batchsize = len(rank_scores)
        batch_ids = [[i for i in range(item_num)] for _ in range(batchsize)]#output_dict["dataset_ids"]  # [batch_size, 100]

        # batchsize = len(rank_scores)/10
        # sample_batch = random.sample(range(batchsize), int(self.test_num))
        sample_batch_ids = []
        seq_len = len(rank_scores[0])  # 100个item
        self.max_len = seq_len
        # ret_labels, ret_cates = [], []

        # batch_MMR_scores = []
        recommended_list = []
        for i in tqdm(range(batchsize)):  # 一个batch 有 n个user
            rank_score = rank_scores[i]
            ids = batch_ids[i]
            sample_batch_ids.append(ids)

            cate_id = {x: M[x] for x in ids}  # 每个id的类别字典
            mask = [0 if i < self.max_len else float('-inf') for i in range(seq_len)]
            # sorted_idx = sorted(range(self.max_len), key=lambda k: rank_score[k], reverse=True)
            first_id = random.choice(ids)
            chosen_list = [first_id]
            for j in range(1, seq_len):
                mmr_score = [mask[k] + lamda * rank_score[k] -
                             (1 - lamda) * (
                                 float('inf') if ids[k] in chosen_list else self.sim(ids[k], chosen_list, cate_id))
                             for k in range(seq_len)]
                sorted_idx = sorted(range(self.max_len),
                                    key=lambda k: mmr_score[k],
                                    reverse=True)

                # ret_label.append(label[sorted_idx[0]])
                # ret_cate.append(cate_id[sorted_idx[0]])
                # chosen_list.append(ids[sorted_idx[0]])
                # mask[sorted_idx[0]] = float('-inf')
                for k in range(0, seq_len):
                    if ids[sorted_idx[k]] not in chosen_list:
                        chosen_list.append(ids[sorted_idx[k]])
                        mask[sorted_idx[k]] = float('-inf')
                        break
            # chosen_list = chosen_list
            recommended_list.append(chosen_list[:self.TopK])
            # chosen_score = {chosen_list[i]: 1 - i / len(chosen_list) for i in range(len(chosen_list))}
            # new_rank_score = [chosen_score[x] for x in ids]
            # batch_MMR_scores.append(new_rank_score)
        return recommended_list
            # chosen_ids = []
            # rank_score = rank_scores[i]
            # ids = batch_ids[i]
            # cate_id = {x: id_multihot[str(x)] for x in ids.tolist()}
            # # sim_scores = self.sim_scores(cate_id)
            # # 计算 MMR score
            # MMR_scors = [lamda*rank_score[j] -  (1-lamda)* self.sim(ids[j],chosen_ids,cate_id) for j in range(len(rank_score))]

        # batch_MMR_scores = np.array(batch_MMR_scores)
        # sample_batch_ids = np.array(sample_batch_ids)
        # return {"predictions": batch_MMR_scores, "dataset_ids": sample_batch_ids}

    # class MMR(object):
#     def __init__(self, max_time_len):
#         self.max_time_len = max_time_len

#     def predict(self, data_batch, lamda=1):
#         cate_ids = list(map(lambda a: [i[1] for i in a], data_batch[2]))
#         rank_scores = data_batch[-1]
#         labels = data_batch[4]
#         seq_lens = data_batch[6]
#         ret_labels, ret_cates = [], []
#         for i in range(len(seq_lens)):
#             ret_label, ret_cate = [], []
#             cate_set = set()
#             cate_id, rank_score, label, seq_len = cate_ids[i], rank_scores[i], labels[i], seq_lens[i]
#             mean_score = sum(rank_score[:seq_len])/seq_len
#             mask = [0 if i < seq_len else float('-inf') for i in range(self.max_time_len)]
#             mmr_score = [rank_score[k] + mask[k] for k in range(self.max_time_len)]
#             sorted_idx = sorted(range(self.max_time_len), key=lambda k:mmr_score[k], reverse=True)
#             mask[sorted_idx[0]] = float('-inf')
#             ret_label.append(label[sorted_idx[0]])
#             ret_cate.append(cate_id[sorted_idx[0]])
#             cate_set.add(cate_id[sorted_idx[0]])
#             for j in range(1, seq_len):
#                 mmr_score = [mask[k] + lamda * rank_score[k] +
#                                         (1 - lamda) * (0 if cate_id[k] in cate_set else abs(mean_score))
#                              for k in range(self.max_time_len)]
#                 sorted_idx = sorted(range(self.max_time_len),
#                                     key=lambda k: mmr_score[k],
#                                     reverse=True)
#                 mask[sorted_idx[0]] = float('-inf')
#                 ret_label.append(label[sorted_idx[0]])
#                 ret_cate.append(cate_id[sorted_idx[0]])
#                 cate_set.add(cate_id[sorted_idx[0]])
#             ret_labels.append(ret_label)
#             ret_cates.append(ret_cate)
#         return ret_labels, ret_cates


