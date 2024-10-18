import pandas as pd
import os
import cvxpy as cp
import numpy as np
import math
from tqdm import tqdm,trange
import yaml
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer
from datetime import datetime



class GPU_layer(nn.Module):
    def __init__(self,  p_size, lambd, rho):
        super(GPU_layer, self).__init__()
        self.rho = rho
        self.A = torch.triu(torch.ones((p_size,p_size)))
        self.d = torch.ones(p_size)
        self.lambd = lambd
        self.p_size = p_size


    def forward(self,x):
        sorted_args = torch.argsort(x*self.rho.to(x.device), dim=-1)
        sorted_x = x.gather(dim=-1, index=sorted_args)

        rho = self.rho.gather(dim=-1, index=sorted_args).cpu()
        answer = cp.Variable(self.p_size)
        para_ordered_tilde_dual = cp.Parameter(self.p_size)
        constraints = []
        constraints += [cp.matmul(cp.multiply(rho, answer),self.A) + self.lambd * self.d >= 0]
        objective = cp.Minimize(cp.sum_squares(cp.multiply(rho,answer) - cp.multiply(rho, para_ordered_tilde_dual)))
        problem = cp.Problem(objective, constraints)
        #assert problem.is_dpp()
        self.cvxpylayer = CvxpyLayer(problem, parameters=[para_ordered_tilde_dual], variables=[answer])

        solution, = self.cvxpylayer(sorted_x)
        re_sort = torch.argsort(sorted_args, dim=-1)
        return solution.to(x.device).gather(dim=-1,index=re_sort)

def compute_projection_maxmin_fairness_with_order(ordered_tilde_dual, rho, lambd):

    m = len(rho)
    answer = cp.Variable(m)
    objective = cp.Minimize(cp.sum_squares(cp.multiply(rho,answer) - cp.multiply(rho, ordered_tilde_dual)))
    #objective = cp.Minimize(cp.sum(cp.multiply(rho,answer) - cp.multiply(rho, ordered_tilde_dual)))
    constraints = []
    for i in range(1, m+1):
        constraints += [cp.sum(cp.multiply(rho[:i],answer[:i])) >= -lambd]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    #print(type(result))
    #exit(0)
    #print(type(answer.value))
    ans = answer.value
    # ans = torch.FloatTensor(ans).to('cuda')
    return ans


def compute_next_dual(eta, rho, dual, gradient, lambd):
    #rho = self.problem.data.rho
    # if isinstance(gradient, torch.Tensor):
    #     gradient = gradient.cpu().numpy()
    #     rho = rho.cpu().numpy()
    #     dual = dual.cpu().numpy()
    tilde_dual = dual - eta*gradient/rho/rho
    order = np.argsort(tilde_dual*rho)

    ordered_tilde_dual = tilde_dual[order]
    # print(order, ordered_tilde_dual)
    # print ordered_tilde_dual*rho
    ordered_next_dual = compute_projection_maxmin_fairness_with_order(ordered_tilde_dual, rho[order], lambd)
    # print(ordered_next_dual)
    # print("tilde_dual", rho*tilde_dual)
    # print("next_dual", rho*ordered_next_dual[order.argsort()])
    return ordered_next_dual[order.argsort()]

class P_MMF(object):
    def __init__(self, TopK, lbd):
        self.device = 'cpu'

        self.TopK = TopK

        self.lambd = lbd #self.hyper_parameters['lambda']
        self.learning_rate = 0.01
        self.gamma = 0.2

    def recommendation(self, batch_UI, M, rho):
        # batch_UI = batch_UI.cpu().numpy()
        batch_UI = torch.FloatTensor(batch_UI).to(self.device)
        self.rho = torch.FloatTensor(rho).to(self.device)
        self.M = torch.FloatTensor(M).to(self.device)
        self.M_sparse = torch.FloatTensor(M).to(self.device).to_sparse()
        batch_size = len(batch_UI)
        B_t = batch_size*self.TopK*self.rho
        self.item_num = batch_UI.shape[1]
        num_providers = self.M.shape[1]
        recommended_list = []
        eta = self.learning_rate/math.sqrt(self.item_num)

        if self.device == 'cuda':
            mu_t = torch.zeros(num_providers).to(self.device)
            gradient_cusum = torch.zeros(num_providers).to(self.device)
            for t in trange(batch_size, desc="user arriving", unit="user"):
                x_title = batch_UI[t, :] - self.M_sparse.matmul(mu_t.t()).t()
                mask = self.M_sparse.matmul((B_t > 0).float().t()).t()

                mask = (1.0 - mask) * -10000.0
                values, items = torch.topk(x_title + mask, k=self.TopK, dim=-1)
                # x = np.argsort(x_title+mask,axis=-1)[::-1]
                #
                re_allocation = torch.argsort(batch_UI[t, items], descending=True)
                x_allocation = items[re_allocation]
                recommended_list.append(x_allocation)
                B_t = B_t - torch.sum(self.M[x_allocation], dim=0, keepdims=False)
                gradient_tidle = -torch.mean(self.M[x_allocation], dim=0, keepdims=False) + B_t / (batch_size * self.TopK)

                gradient = self.gamma * gradient_tidle + (1 - self.gamma) * gradient_cusum
                gradient_cusum = gradient

                for g in range(1):
                    mu_t = self.update_mu_func(mu_t - eta * gradient / self.rho / self.rho)
                    # mu_t = compute_next_dual(eta, self.rho, mu_t, gradient, self.lambd)
            return recommended_list
        else:
            batch_UI = batch_UI.cpu().numpy()
            self.M = self.M.cpu().numpy()
            self.rho = self.rho.cpu().numpy()
            B_t = B_t.cpu().numpy()
            mu_t = np.zeros(num_providers)
            gradient_cusum = np.zeros(num_providers)
            for t in trange(batch_size):

                x_title = batch_UI[t, :] - np.matmul(self.M, mu_t)
                mask = np.matmul(self.M, (B_t>0).astype(np.float))
                mask = (1.0-mask) * -10000.0
                x = np.argsort(x_title+mask,axis=-1)[::-1]
                x_allocation = x[:self.TopK]
                re_allocation = np.argsort(batch_UI[t,x_allocation])[::-1]
                x_allocation = x_allocation[re_allocation]
                recommended_list.append(x_allocation)
                B_t = B_t - np.sum(self.M[x_allocation],axis=0,keepdims=False)
                gradient = -np.mean(self.M[x_allocation],axis=0,keepdims=False) + self.rho

                #gradient_list.append(gradient)
                gradient = self.gamma * gradient + (1-self.gamma) * gradient_cusum
                gradient_cusum = gradient
                #gradient = -(B_0-B_t)/((t+1)*K) + rho
                for g in range(1):
                    mu_t = compute_next_dual(eta, self.rho, mu_t, gradient, self.lambd)
            return recommended_list


