import pandas as pd
import os
import cvxpy as cp
import numpy as np
import math
from tqdm import tqdm,trange
import yaml
import torch



class FairCo(object):
    def __init__(self, TopK, lbd):

        self.TopK = TopK

        self.lambd = lbd

    def recommendation(self, batch_UI, M ,rho):
        batch_UI = batch_UI#.cpu().numpy()
        self.M = M
        self.rho = rho
        self.item_num = M.shape[0]
        batch_size = len(batch_UI)
        B_u = batch_size*self.TopK*self.rho

        _ , num_providers = self.M.shape
        B_l = np.zeros(num_providers)
        recommended_list = []

        for t in range(batch_size):
            mask = (B_u>0).astype(np.float)
            mask = np.matmul(self.M,mask)
            recommended_mask = (1-mask) * -10000.0
            minimax_reg = self.lambd * np.matmul(self.M,(-B_l + np.min(B_l))/(self.rho * batch_size))
            rel = batch_UI[t,:] + minimax_reg
            result_item = np.argsort(rel + recommended_mask)[::-1]
            result_item = result_item[:self.TopK]
            vec = batch_UI[t,:]
            sorted_index = np.argsort(vec[result_item])[::-1]
            recommended_list.append(result_item[sorted_index])
            B_u = B_u - np.sum(self.M[result_item,:],axis=0,keepdims=False)
            B_l = B_l + np.sum(self.M[result_item,:],axis=0,keepdims=False)

        return recommended_list