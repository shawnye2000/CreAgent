import math

import numpy as np
import heapq
import os
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time

from numpy import mean

# from librerank.utils import *
# from librerank.reranker import *
# from librerank.rl_reranker import *
import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from collections import defaultdict
class APDR(object):
    def __init__(self, TopK, lbd):
        self.TopK = TopK
        self.lamda = lbd
    def div_score(self, cate_id, mp):
        return math.log2(2 + mp[cate_id]) - math.log2(1 + mp[cate_id])

    def recommendation(self, UI_matrix, M, rho):
        user_num = UI_matrix.shape[0]
        item_num = UI_matrix.shape[1]
        cate_ids = [[np.where(row == 1)[0][0] for row in M] for u in range(user_num)]
        seq_lens = [item_num for u in range(user_num)]
        recommend_list = []
        for i in range(len(seq_lens)):  # 第i个user
            topk_list = []
            cate_mp = defaultdict(int)
            cate_id, seq_len = cate_ids[i], seq_lens[i]
            # mean_score = sum(rank_score[:seq_len]) / seq_len
            mean_score = 1
            mask = [0 if k < seq_len else float('-inf') for k in range(item_num)]
            pred_score = [UI_matrix[i][k] + mask[k] for k in range(item_num)]
            sorted_idx = sorted(range(item_num), key=lambda k: pred_score[k], reverse=True)
            mask[sorted_idx[0]] = float('-inf')
            cate_mp[cate_id[sorted_idx[0]]] += 1
            topk_list.append(sorted_idx[0])
            for j in range(1, seq_len):
                pred_score = [mask[k] + self.lamda * UI_matrix[i][k] +
                              (1 - self.lamda) * abs(mean_score) * self.div_score(cate_id[k], cate_mp)
                              for k in range(item_num)]
                sorted_idx = sorted(range(item_num),
                                    key=lambda k: pred_score[k],
                                    reverse=True)
                mask[sorted_idx[0]] = float('-inf')
                cate_mp[cate_id[sorted_idx[0]]] += 1
                topk_list.append(sorted_idx[0])
            recommend_list.append(topk_list[:self.TopK])
        return recommend_list
