# -*- coding: utf-8 -*-
import os
import pandas
import numpy as np
import random
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import torch.nn as nn
from scipy.stats import rankdata
import pprint
from utils import *
from datetime import datetime
from torchmetrics import PearsonCorrCoef
random.seed(2021)

#similarity_fun=['cosine','correlation','euclidean']

def cosine_distance_torch(x1, eps=1e-10):
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x1.t()) / (w1 * w1.t()).clamp(min=eps)

def correlation_torch(x):
    flag = False
    if torch.is_tensor(x)==False:
        x=torch.from_numpy(x)
        flag=True
    corr=torch.corrcoef(x)
    if flag==True:
        corr=corr.numpy()
    return corr

def euclidean_similarity(x):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    flag = False
    if torch.is_tensor(x) == False:
        x = torch.from_numpy(x)
        flag = True
    y=x
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    maximum=dist.max()
    similarity=maximum-dist
    if flag==True:
        similarity=similarity.numpy()
    return similarity


def PageRank_matrix(P, c=0.15):
    """
    param P: transfer probability matrix
    """
    n,_ = np.shape(P)
    s = np.zeros(n)
    while np.sum(abs(s - ((1 - c) * np.matmul(P,s) + c/n))) > 0.0000001:
        s = (1 - c) * np.matmul(P,s) + c/n
    return s

def get_connectivity_scores(features,similarity_fun,thre):
    if similarity_fun=='cosine':
        similarity_matrix = cosine_distance_torch(features)
    if similarity_fun == 'correlation':
        similarity_matrix = correlation_torch(features)
    if similarity_fun == 'euclidean':
        similarity_matrix = euclidean_similarity(features)
        #thre = torch.median(similarity_matrix).cpu().numpy()
        #thre= torch.quantile(torch.reshape(similarity_matrix, (-1,)), 0.6)
    similarity_matrix=similarity_matrix.cpu().detach().numpy()
    #(np.round(similarity_matrix, 3) != np.round(similarity_matrix, 3).transpose()).sum()
    similarity_matrix=(similarity_matrix>thre).astype(int)*similarity_matrix
    normalized_transition_matrix= np.divide(similarity_matrix, np.sum(similarity_matrix,axis=1))
    connectivity_scores=PageRank_matrix(normalized_transition_matrix, c=0.15)
    return similarity_matrix,normalized_transition_matrix,connectivity_scores

def sort_features_by_index(sorted_index,features,connectivity_scores,similarity_matrix,labels):
    features = features[sorted_index]
    connectivity_scores = connectivity_scores[sorted_index]
    similarity_matrix = similarity_matrix[sorted_index, :]
    similarity_matrix = similarity_matrix[:, sorted_index]
    labels = labels[sorted_index]
    #np.where(sorted_index == 0)
    # matrix=np.array([[1,0,0,0,0],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,1],[0,0,0,0,1]])
    # index=np.array([3,0,2,1,4])
    # matrix=matrix[index,:]
    # matrix=matrix[:,index]
    return features,connectivity_scores,similarity_matrix,labels

def constraint_loss(sumB,K):
    '''sumB is the sum of B'''
    loss = torch.abs(sumB - K)#add constraint to puch sum(B) near to K
    return loss

'''Attacker optimization'''
class ConstraintProjector(object):
    '''projected gradient descend'''
    def __init__(self,dataset):
        print("projector initiate")
        if dataset=="KDD99":
            f = open(r"../DataSets/NetworkIntrusion/KDD99/data_constraint_info.pkl", 'rb')
            self.content = pickle.load(f)
            f.close()
            self.feature_index = self.content[0]
            self.feature_range = self.content[1]
            self.variable_types = self.content[2]
            print("constraints of features:")
            pprint.pprint(self.feature_range)
    def __call__(self, module):
        if hasattr(module, 'control_features'):
            print("apply Constraint projection")
            w = module.control_features.data
            o = module.original_control_features.data

            for v in self.variable_types["continous_variables"]+self.variable_types["integer_variables"]:
                constraint = self.feature_range[v]
                index = self.feature_index[v]
                w[:,index] = w[:,index].clamp(constraint[0], constraint[1])

            for i in range(1,14):
                w[:, i] = o[:, i]
                #print("reset original")
            module.control_features.data = w

class Attacker_optimization(nn.Module):
    '''Attacker optimization: get the best solution to attack'''
    def __init__(self,b, features,connectivity_scores,c,targets,similarity_fun,controlled_nodes,threshold,args,atk_graph=None):
        super(Attacker_optimization, self).__init__()
        self.device=args.device
        self.args=args
        n=features.shape[0]
        self.I = torch.eye(n).to(self.device)
        self.ones = torch.ones(n).to(self.device)
        self.budget=controlled_nodes.shape[0]
        self.c=torch.FloatTensor([c]).to(self.device)
        self.n=torch.FloatTensor([n]).to(self.device)
        self.target_nodes=torch.LongTensor(targets).to(self.device)
        self.other_ndoes=np.setdiff1d(range(0, n),targets)
        self.control_features = torch.nn.Parameter(features.clone()[0:self.budget].to(self.device))
        self.other_features = features.clone()[self.budget:].to(self.device)
        self.all_features = features.clone().to(self.device)
        self.original_control_features = features.clone()[0:self.budget].to(self.device)
        self.control_features.requires_grad = True
        #self.connectivity_scores = torch.zeros(n).to(self.device)
        self.connectivity_scores = torch.FloatTensor(connectivity_scores).to(self.device)
        self.similarity_fun=similarity_fun
        self.thre = nn.Threshold(threshold, 0.0)
        self.opt = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.lamda)
        if self.args.apply_constraint:
            self.projector = ConstraintProjector(self.args.dataset)
        if self.args.attack_loss == "attacked_graph":
            self.atk_edges, self.atk_target_similarity, atk_edges_num= atk_graph
            self.atk_target_similarity=torch.clamp(self.atk_target_similarity,min=threshold-0.1)
            self.cos = torch.nn.CosineSimilarity(dim=0)
            self.cor = PearsonCorrCoef(num_outputs=atk_edges_num).to(self.device)


    def get_connectivity_scores(self,features):
        if self.similarity_fun == 'cosine':
            similarity_matrix = cosine_distance_torch(features)
        if self.similarity_fun == 'correlation':
            similarity_matrix = correlation_torch(features)
        if self.similarity_fun == 'euclidean':
            similarity_matrix = euclidean_similarity(features)
        similarity_matrix = self.thre(similarity_matrix)
        similarity_matrix = similarity_matrix.cpu().detach().numpy()
        normalized_transition_matrix = np.divide(similarity_matrix, np.sum(similarity_matrix, axis=1))
        connectivity_scores = PageRank_matrix(normalized_transition_matrix, c=0.15)
        return similarity_matrix, normalized_transition_matrix, connectivity_scores

    def atk_loss(self,connectivity_scores):
        if self.args.attack_loss=='target_anomaly_adaptive':# adaptive loss
            with torch.no_grad():
                alpha = torch.zeros(self.args.target_node_num).to(self.device)
                rankings = rankdata(connectivity_scores.cpu().numpy())  # 返回 ranking
                for i,node in enumerate(self.target_nodes):
                    rank = rankings[node]
                    alpha[i] = self.mapping_function(rank, k=120)
                print(alpha)
        elif self.args.attack_loss=='target_anomaly':
            alpha = torch.ones(self.args.target_node_num).to(self.device)
        elif self.args.attack_loss=="attacked_graph":
            alpha = torch.ones(self.args.target_node_num).to(self.device)
            if self.similarity_fun=="cosine":
                return torch.square(self.atk_target_similarity-self.cos(self.all_features[self.atk_edges[0]].t(), self.control_features[self.atk_edges[1]].t())).sum()
            else:
                return torch.square(self.atk_target_similarity-self.cor(self.all_features[self.atk_edges[0]].t(), self.control_features[self.atk_edges[1]].t())).sum()
        return -torch.sum(connectivity_scores[self.target_nodes]*alpha)*1000

    def gaussian(self, x,mu=1.0, sigma=500.0):
        return torch.exp(-torch.pow(x - mu, 2.) / (2 * torch.pow(sigma, 2.)))

    def mapping_function(self,x,k=50):
        # k=600
        # x = torch.arange(0, 2000, 1.0)
        # y = []
        # for t in x:
        #     if t<=k:
        #         y.append(self.gaussian(t, mu=k, sigma=torch.tensor(500.0)))
        #     else:
        #         y.append(0.0)
        # plt.plot(x.numpy(),y)
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.show()
        if x<=k:
            y = 100*self.gaussian(torch.tensor(x), mu=k, sigma=torch.tensor(k/2))
        else:
            y = 0.0
        return y

    def forward(self):
        forward_features=torch.concat((self.control_features,self.other_features),dim=0)
        if self.similarity_fun == 'cosine':
            similarity_matrix = cosine_distance_torch(forward_features)
        if self.similarity_fun == 'correlation':
            similarity_matrix = correlation_torch(forward_features)
        if self.similarity_fun == 'euclidean':
            similarity_matrix = euclidean_similarity(forward_features)
        similarity_matrix = self.thre(similarity_matrix)
        self.normalized_transition_matrix = torch.div(similarity_matrix, similarity_matrix.sum(1))
        self.connectivity_scores = (1 - self.c) * torch.matmul(self.normalized_transition_matrix, self.connectivity_scores.detach()) + self.c / self.n
        return self.connectivity_scores

    def closed_form_forward(self):
        forward_features = torch.concat((self.control_features, self.other_features), dim=0)
        if self.similarity_fun == 'cosine':
            similarity_matrix = cosine_distance_torch(forward_features)
        if self.similarity_fun == 'correlation':
            similarity_matrix = correlation_torch(forward_features)
        if self.similarity_fun == 'euclidean':
            similarity_matrix = euclidean_similarity(forward_features)
        similarity_matrix = self.thre(similarity_matrix)
        self.normalized_transition_matrix = torch.div(similarity_matrix, similarity_matrix.sum(1))
        self.connectivity_scores = self.c * torch.matmul(torch.inverse(self.I -(1-self.c)*self.normalized_transition_matrix), self.ones*self.c / self.n)
        return self.connectivity_scores

    def attack(self,results_data):
        min_loss = 1000
        start = datetime.now()
        for i in range(self.args.attack_epoch):
            self.opt.zero_grad()
            if self.args.attack_mode.split('_')[-1] =="cf":
                connectivity_scores = self.closed_form_forward()
            else:
                connectivity_scores = self.forward()
            loss = self.atk_loss(connectivity_scores)
            print(i, 'loss:', loss.item())
            #torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            loss.backward(retain_graph=True)
            for name, param in self.named_parameters():
                if torch.isnan(param.grad).any():
                    print("nan gradient found")
            self.opt.step()
            torch.cuda.empty_cache()
            with torch.no_grad():
                if self.args.apply_constraint:
                    self.apply(self.projector)
                if self.args.save_opt and loss < min_loss and i > 1:
                    min_loss = loss
                    print("save model")
                    torch.save(self.state_dict(), self.args.output_dir + 'model.pth')
        print('time consuming for attack:', datetime.now() - start)

        '''Get the optimal feature'''
        if self.args.save_opt:
            self.load_state_dict(torch.load(self.args.output_dir + 'model.pth'))
            print("Model's state_dict:")
            for param_tensor in self.state_dict():
                print(param_tensor, "\t", self.state_dict()[param_tensor].size())
            control_features = self.state_dict()[param_tensor]
        else:
            control_features = self.control_features.detach()

        if self.args.apply_constraint:
            print("set integer variables")
            for v in self.projector.variable_types["integer_variables"]:
                constraint = self.projector.feature_range[v]
                index = self.projector.feature_index[v]
                control_features[:, index] = control_features[:, index].clone().type(torch.int)
                control_features[:,index] = control_features[:,index].clamp(constraint[0], constraint[1])

        features_atk=torch.concat((control_features, self.other_features), dim=0)
        similarity_matrix_atk, _, connectivity_scores_atk = self.get_connectivity_scores(features_atk)
        targets_scores = connectivity_scores_atk[self.target_nodes.cpu().numpy()]
        others_scores = connectivity_scores_atk[self.other_ndoes]
        print('***Attacked targets connectivity scores')
        print('maximum:', np.min(targets_scores))
        print('avarage:', np.mean(targets_scores))
        ranks = rankdata(connectivity_scores_atk)
        print(ranks[self.target_nodes.cpu().numpy()])
        results_data['anomaly score'].append(1 - connectivity_scores_atk[self.target_nodes.cpu().numpy()])
        results_data['ranking'].append(ranks[self.target_nodes.cpu().numpy()].mean() / self.args.data_size)
        results_data['detect top1%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.01))
        results_data['detect top5%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.05))
        results_data['detect top10%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.1))
        results_data['similarity'].append(self.similarity_fun)
        return connectivity_scores_atk,similarity_matrix_atk,features_atk,results_data
