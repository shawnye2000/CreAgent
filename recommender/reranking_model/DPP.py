import numpy as np
import heapq
import os
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time
import math
import torch
from numpy import mean
from tqdm import tqdm

import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


class DPP(object):
    def __init__(self, TopK, lbd):
        self.TopK = TopK
        self.lamda = lbd
    """
    	output_dict = self.predict(dataset) ### {"predictions":predictions([2874（all) x 100]),"dataset_ids":dataset_ids([2874（all) x 100])} 
		baseline = MMR()
		output_dict = baseline.post_process(output_dict, id_multihot,lamda=1)
    """

    def recommendation(self, UI_matrix, M, rho):
        """
        fast implementation of the greedy algorithm
        :param kernel_matrix: 2-d array
        :param max_length: positive int
        :param epsilon: small positive scalar
        :return: list
        """
        epsilon = 1E-10
        item_size = UI_matrix.shape[1]
        max_length = item_size
        cate_size = M.shape[1]
        embedding_layer = torch.nn.Embedding(cate_size, 5000)

        embed_list = []

        for row in M:
            index = np.where(row == 1)[0]
            category_index = torch.tensor([index])
            embedding = embedding_layer(category_index)
            embed_list.append(embedding.tolist())

        recommend_list = []
        for i in range(UI_matrix.shape[0]):
            scores = UI_matrix[i]

            # 示例 one-hot 向量对应的类别索引，假设类别是2

            feature_vectors = np.array(embed_list).squeeze()  #np.random.randn(item_size, feature_dimension)  # shpae (item_size,dim)#商品embeding

            feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)  # feature除以每个行向量的2范数（也就是行向量的模）
            similarities = np.dot(feature_vectors, feature_vectors.T)
            kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))  # k核矩阵

            item_size = kernel_matrix.shape[0]
            cis = np.zeros((max_length, item_size))
            di2s = np.copy(np.diag(kernel_matrix))  # shape为(item_size,)
            selected_items = list()
            selected_item = np.argmax(di2s)
            selected_items.append(selected_item)
            while len(selected_items) < max_length:
                k = len(selected_items) - 1
                ci_optimal = cis[:k, selected_item]
                di_optimal = math.sqrt(di2s[selected_item])
                elements = kernel_matrix[selected_item, :]
                eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
                cis[k, :] = eis
                di2s -= np.square(eis)
                di2s[selected_item] = -np.inf
                selected_item = np.argmax(di2s)
                if di2s[selected_item] < epsilon:
                    break
                selected_items.append(selected_item)
            recommend_list.append(selected_items[:self.TopK])
        return recommend_list
