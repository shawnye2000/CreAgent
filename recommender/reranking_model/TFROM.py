import pandas as pd
import numpy as np
import csv
import math
import yaml
from tqdm import trange

class TFROM(object):
    def __init__(self, TopK, lbd):
        # self.M = M
        self.TopK = TopK
        # self.item_num = item_num
        # self.provider_num = M.shape[1]
        # f = open("properties/TFROM.yaml",'r')
        # self.hyper_parameters = yaml.load(f)
        # f.close()

    def recommendation(self, batch_UI, M, rho):
        self.M = M
        self.item_num = batch_UI.shape[1]

        item_num, provider_num = self.M.shape

        # 获取 item_id 列
        item_id = np.arange(item_num)

        # 获取 item 属于的 provider id 列
        provider_id = np.argmax(self.M, axis=1)

        # 获取 provider 有多少个 item 列
        provider_item_count = np.sum(self.M, axis=0)

        item_provider = np.zeros((item_num, 3),dtype=int)
        for i in range(item_num):
            item_provider[i][0] = item_id[i]
            item_provider[i][1] = provider_id[i]
            item_provider[i][2] = provider_item_count[provider_id[i]]
        # print(item_provider)
        # 创建最终的 item 行 3 列矩阵
        # item_provider = np.column_stack((item_id, provider_id, provider_item_count))
        score = batch_UI  #.cpu().numpy()
        sorted_score = []
        for i in range(len(score)):
            sorted_score.append(np.argsort(-score[i]))
        total_round = batch_UI.shape[0]
        provider_exposure_score = [0 for i in range(provider_num)]
        recommended_list = []
        for round_temp in trange(total_round):
            total_exposure = 0
            for i in range(self.TopK):
                total_exposure += 1 / math.log((i + 2), 2)
            total_exposure = total_exposure * (round_temp + 1)

            fair_exposure = []
            for i in range(provider_num):
                fair_exposure.append(total_exposure / self.item_num * provider_item_count[i])

            # next_user = random_user[round_temp]
            # next_user = batch_UI[round_temp]
            next_user = round_temp
            rec_flag = list(sorted_score[next_user])
            rec_result = [-1 for i in range(self.TopK)]

            # find next item and provider
            for top_k in range(self.TopK):
                for next_item in rec_flag:
                    next_provider = item_provider[next_item][1]

                    if provider_exposure_score[next_provider] + 1 / math.log((top_k + 2), 2) <= fair_exposure[
                        next_provider]:
                        rec_result[top_k] = next_item
                        provider_exposure_score[next_provider] += 1 / math.log((top_k + 2), 2)
                        rec_flag.remove(next_item)
                        break
                # print('round:%d, rank:%d' % (round_temp, top_k))

            for top_k in range(self.TopK):
                if rec_result[top_k] == -1:
                    next_item = rec_flag[0]
                    next_provider = item_provider[next_item][1]
                    rec_result[top_k] = next_item
                    provider_exposure_score[next_provider] += 1 / math.log((top_k + 2), 2)
                    del rec_flag[0]

            recommended_list.append(rec_result)
        return recommended_list
