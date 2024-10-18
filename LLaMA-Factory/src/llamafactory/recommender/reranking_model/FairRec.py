import pandas as pd
import os
import numpy as np
import math
from tqdm import tqdm,trange
import yaml
import torch

def greedy_round_robin(m,n,R,l,T,V,U,F):
    # greedy round robin allocation based on a specific ordering of customers (assuming the ordering is done in the relevance scoring matrix before passing it here)

    # creating empty allocations
    B = {}
    for u in U:
        B[u] = []

    # available number of copies of each producer
    Z= {} # total availability
    P = range(n) # set of producers
    for p in P:
        Z[p] = l

    # allocating the producers to customers
    for t in trange(1, R+1):
        #print("GRR round number==============================",t)
        for i in trange(m):
            if T == 0:
                return B,F
            u = U[i]
            # choosing the p_ which is available and also in feasible set for the user
            possible = [(Z[p]>0)*(p in F[u])*V[u,p] for p in range(n)]
            p_ = np.argmax(possible)

            if (Z[p_]>0) and (p_ in F[u]) and len(F[u])>0:
                B[u].append(p_)
                F[u].remove(p_)
                Z[p_] = Z[p_]-1
                T = T-1
            else:
                return B , F
    # returning the allocation
    return B,F


def FairRec_train(U,P,k,V,alpha, m , n):
    # Allocation set for each customer, initially it is set to empty set
    A = {}
    for u in U:
        A[u] = []

    # feasible set for each customer, initially it is set to P
    F = {}
    for u in U:
        F[u]=P[:]
    #print(sum([len(F[u]) for u in U]))

    # number of copies of each producer
    l=int(alpha*m*k/(n+0.0))

    # R = number of rounds of allocation to be done in first GRR
    R = int(math.ceil((l*n)/(m+0.0)))


    # total number of copies to be allocated
    T = l*n

    # first greedy round-robin allocation
    [B,F1]=greedy_round_robin(m,n,R,l,T,V,U[:],F.copy())
    F={}
    F=F1.copy()
    #print("GRR done")
    # adding the allocation
    for u in U:
        A[u]=A[u][:]+B[u][:]

    # second phase
    u_less=[] # customers allocated with <k products till now
    for u in A:
        if len(A[u])<k:
            u_less.append(u)

    # allocating every customer till k products
    for u in u_less:
        scores = V[u,:]
        new=scores.argsort()[-(k+k):][::-1]
        for p in new:
            if p not in A[u]:
                A[u].append(p)
            if len(A[u])==k:
                break

    return A

class FairRec(object):
    def __init__(self, TopK, para):
        #self.rho = rho
        # self.M = M
        self.TopK = TopK
        # self.item_num = item_num
        # f = open("properties/FairRec.yaml",'r')
        # self.hyper_parameters = yaml.load(f)
        # f.close()

        self.para = para #self.hyper_parameters['para']

    def recommendation(self, batch_UI, M, rho):
        self.M = M
        # batch_UI = batch_UI.cpu().numpy()
        batch_size = len(batch_UI)
        provider_num = self.M.shape[1]
        self.item_num = self.M.shape[0]
        U=list(range(batch_size)) # list of customers
        P=list(range(self.item_num)) # list of producers
        # P = list(range(provider_num))
        # U2P_matrix = np.matmul(batch_UI, self.M) / (np.sum(self.M, axis=0, keepdims=True))
        #alpha = rho
        # recommendation_list = FairRec_train(U, P, self.TopK, batch_UI, alpha=self.para, m=batch_size, n=self.item_num)
        recommendation_list = FairRec_train(U, P, self.TopK, batch_UI, alpha=self.para, m=batch_size, n=provider_num)
        return list(recommendation_list.values())

        #rho_reverse = 1/(self.rho*batch_size*self.TopK)



