# -*- coding: utf-8 -*-
import os
import pandas
import numpy as np
import random
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from datetime import datetime
import torch.nn as nn
from scipy.stats import rankdata
from scipy import sparse
from utils import *
from DeepWalkAttack.node_embedding_attack.perturbation_attack import *
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
    dist = torch.cdist(x, x, p=2)  # for numerical stability
    similarity=torch.exp(-dist/100)
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
    similarity_matrix=(similarity_matrix>thre).astype(int)*similarity_matrix
    normalized_transition_matrix= np.divide(similarity_matrix, np.sum(similarity_matrix,axis=1))
    connectivity_scores=PageRank_matrix(normalized_transition_matrix, c=0.15)
    return similarity_matrix,normalized_transition_matrix,connectivity_scores

def lasso_loss(loss1):
    '''loss1 is the original loss of model'''
    all_params = torch.cat([param.view(-1) for param in attacker.parameters()])
    l1_regularization = lamda * torch.norm(all_params, 1)
    loss = loss1 + l1_regularization  # lasso
    return loss

def constraint_loss(sumB):
    '''sumB is the sum of B'''
    loss = torch.abs(sumB)
    return loss

'''Attacker optimization'''
class ConstraintProjector(object):
    '''projected gradient descend'''

    def __init__(self,K,scaling=False):
        self.K=K
        self.scaling=scaling

    def __call__(self, module):
        if hasattr(module, 'B'):
            #print("apply Constraint projection")
            w = module.B.data
            w = w.clamp(0, 1)  # projected B to [0,1]
            if self.scaling:
                total = w.sum()
                if total > 1000:
                    w = w * (self.K*10 / total)  # scale the B as: sum(B)=K #nan if total=0.0
                    print('scaling B')
            module.B.data = w


class Attacker_optimization(nn.Module):
    '''Attacker optimization: get the best solution to attack'''
    def __init__(self, features,c,similarity_matrix,connectivity_scores,target_nodes,similarity_fun,thre,args):
        super(Attacker_optimization, self).__init__()
        self.device = args.device
        self.args = args
        self.c = torch.FloatTensor([c]).to(self.device)
        self.n = features.shape[0]# number of node
        self.n0 = args.target_node_num
        self.similarity_matrix = torch.FloatTensor(similarity_matrix).to(self.device)
        self.total_edges = (similarity_matrix > 0).sum()
        self.target_nodes = torch.LongTensor(np.array(target_nodes)).to(self.device)
        self.connectivity_scores = torch.FloatTensor(connectivity_scores).to(self.device)
        self.similarity_fun = similarity_fun
        self.thre=thre
        self.t = nn.Threshold(thre, 0.0)
        self.B = nn.Parameter(torch.Tensor(self.n, self.n))
        if args.attack_mode=='cf-greedy':
            self.B.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.B)
        self.B.requires_grad = True
        self.I=torch.eye(self.n).to(self.device)
        self.ones=torch.ones(self.n).to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.lamda)

    def symmetric(self,X):#the diag part cannot changed, and the changes are sysmmetric, since the similarity function is sysmmetric
        return X.triu(1) + X.triu(1).transpose(-1, -2)

    def clip_gradient(self, optimizer, grad_clip):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.
        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def symmetric(self,X):  # the diag part cannot changed, and the changes are sysmmetric, since the similarity function is sysmmetric
        return X.triu(1) + X.triu(1).transpose(-1, -2)

    def update_connectivity_scores(self):
        P=torch.abs(self.symmetric(self.B) - self.similarity_matrix)
        self.normalized_P = torch.div(P, P.sum(1))
        self.connectivity_scores = (1 - self.c) * torch.matmul(self.normalized_P, self.connectivity_scores.detach()) + self.c / self.n
        return self.connectivity_scores

    def closed_form_forward(self):
        P = torch.abs(self.symmetric(self.B) - self.similarity_matrix)
        self.normalized_P = torch.div(P, P.sum(1))
        self.connectivity_scores = self.c * torch.matmul(torch.inverse(self.I -(1-self.c)*self.normalized_P), self.ones*self.c / self.n)
        return self.connectivity_scores

    def closed_form(self,B):
        similarity_matrix = torch.abs(self.symmetric(B) - self.similarity_matrix)
        similarity_matrix = (similarity_matrix > self.thre) * similarity_matrix
        P = torch.div(similarity_matrix, similarity_matrix.sum(1))
        connectivity_scores = self.c * torch.matmul(torch.inverse(self.I - (1 - self.c) * P),
                                                         self.ones * self.c / self.n)
        return connectivity_scores

    def cfGreedy_forward(self):
        P = self.symmetric(self.B) + self.similarity_matrix
        self.normalized_P = torch.div(P, P.sum(1))
        self.connectivity_scores = self.c * torch.matmul(torch.inverse(self.I -(1-self.c)*self.normalized_P), self.ones*self.c / self.n)
        #self.connectivity_scores = self.c * torch.inverse(self.I - (1 - self.c) * self.normalized_P) * self.c / self.n
        return self.connectivity_scores

    def atk_loss(self,connectivity_scores):#attack loss function
        if self.args.attack_loss=='target_anomaly_adaptive':# adaptive loss
            with torch.no_grad():
                alpha = torch.zeros(self.args.target_node_num).to(self.device)
                rankings = rankdata(connectivity_scores.cpu().numpy())  # 返回 ranking
                for i,node in enumerate(self.target_nodes):
                    rank = rankings[node]
                    alpha[i] = self.mapping_function(rank, k=4000)
                print(alpha)
        elif self.args.attack_loss=='target_anomaly_sum':
            alpha = torch.ones(self.args.target_node_num).to(self.device)
        return -(torch.sum(connectivity_scores[self.target_nodes]*alpha)/torch.sum(connectivity_scores))*1000

    def mapping_function(self, x, k=50):
        if x <= k:
            y = 1
        else:
            y = 0.1
        return y

    def forward(self):
        connectivity_scores=self.update_connectivity_scores()
        return connectivity_scores

    def attack(self,budget):
        loss_data = {'epochs': [], 'attack loss': [],'attack loss approximate':[],'attack loss discrete':[]}#,'anomaly scores':[],'similarity':[]
        K = budget
        projector = ConstraintProjector(K,self.args.scaling)
        min_loss = 1000
        start = datetime.now()
        for i in range(self.args.attack_epoch):
            self.opt.zero_grad()
            if self.args.attack_mode== 'alternative':
                connectivity_scores = self.forward()
            elif self.args.attack_mode== 'closed-form':
                connectivity_scores = self.closed_form_forward()
            loss1 = self.atk_loss(connectivity_scores)
            for param in self.parameters():
                sumB = param.sum()  # This parameters are attack matrix B
            loss = loss1 + self.args.lamda * constraint_loss(sumB)
            print(i, 'loss:', loss1.item())
            loss_data['epochs'].append(i)
            loss_data['attack loss approximate'].append(loss1.item())
            with torch.no_grad():
                connectivity_scores_real = self.closed_form(self.B)
                loss_real=self.atk_loss(connectivity_scores_real)
                loss_data['attack loss'].append(loss_real.item())

            loss.backward(retain_graph=False)
            #self.clip_gradient(self.opt, 1.0)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            for name, param in self.named_parameters():
                if torch.isnan(param.grad).any():
                    print("nan gradient found")
            self.opt.step()

            with torch.no_grad():
                self.apply(projector)
            if self.args.save_B_opt_loss and loss1 < min_loss and i > 1:
                min_loss = loss1
                print("save model")
                torch.save(self.state_dict(), self.args.output_dir + 'model.pth')

            if self.args.attack_epoch>120:
                with torch.no_grad():
                    B=self.B.detach()
                    topk_index = largest_indices(B.triu(), 200)
                    topk_index = [list(topk_index[0]), list(topk_index[1])]
                    topk_B = torch.zeros(self.n, self.n).to(self.device)
                    topk_B[topk_index[0], topk_index[1]] = B[
                        torch.tensor(topk_index[0]).to(self.device), torch.tensor(topk_index[1]).to(self.device)]
                    '''Show the results'''
                    connectivity_scores_atk = self.closed_form(topk_B)
                    loss_discrete=-(torch.sum(connectivity_scores_atk[self.target_nodes]) / torch.sum(connectivity_scores_atk)) * 1000
                    loss_data['attack loss discrete'].append(loss_discrete.item())

        f = open(f'{self.args.output_dir}/loss_data_{self.similarity_fun}.pkl', 'wb')
        pickle.dump([loss_data], f)
        f.close()
        pprint.pprint(loss_data)
        print('time consuming for attack:', datetime.now() - start)

    def attack_greedy(self,budget,average_degree,results_data): #closed-form greedy attack
        start = datetime.now()
        total_budgets=[]
        for b in budget:
            if self.args.budget_mode == "totoal_edges":
                total_budget = int(total_edges * b)
            else:
                total_budget = int(average_degree * b * self.args.target_node_num)
            total_budgets.append(total_budget)
        for i in range(total_budgets[-1]):
            connectivity_scores = self.cfGreedy_forward()
            loss1 = self.atk_loss(connectivity_scores)
            print(i, 'loss:', loss1.item())
            B_grad = torch.autograd.grad(loss1, self.B, retain_graph=True)[0]
            mask=(((B_grad>0) & (self.similarity_matrix!=0)) | ((B_grad<0) & (self.similarity_matrix!=1)))& (self.B==0)
            B_grad=mask*B_grad
            max_index = largest_indices(torch.abs(B_grad).triu(), 1)
            print('max B_grad:',B_grad[max_index].item())
            if B_grad[max_index]>0 and self.similarity_matrix[max_index] !=0:
                self.B.data[max_index] = -self.similarity_matrix[max_index]
            elif B_grad[max_index]<0 and self.similarity_matrix[max_index] !=1:
                self.B.data[max_index] = 1.0-self.similarity_matrix[max_index]
            else:
                print("exception")
            if i in total_budgets:
                print(f"total budget: {i}")
                with torch.no_grad():
                    similarity_matrix = self.symmetric(self.B.data) + self.similarity_matrix
                    print("min and max of attacked similarity_matrix:", similarity_matrix.min().item(), similarity_matrix.max().item())
                    similarity_matrix = similarity_matrix.cpu().detach().numpy()
                    similarity_matrix = (similarity_matrix > self.thre).astype(int) * similarity_matrix
                    normalized_transition_matrix = np.divide(similarity_matrix, np.sum(similarity_matrix, axis=1))
                    connectivity_scores_atk = PageRank_matrix(normalized_transition_matrix, c=0.15)
                    targets_scores = [connectivity_scores_atk[t] for t in self.target_nodes]
                    print('***Attacked targets connectivity scores')
                    print('minimum:', np.min(targets_scores))
                    print('avarage:', np.mean(targets_scores))
                    ranks = rankdata(connectivity_scores_atk)
                    results_data['anomaly score'].append((1 - connectivity_scores_atk)[self.target_nodes.cpu().numpy()])
                    results_data['ranking'].append(ranks[self.target_nodes.cpu().numpy()].mean() / self.args.data_size)
                    results_data['detect top1%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.01))
                    results_data['detect top5%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.05))
                    results_data['detect top10%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.1))
                    results_data['similarity'].append(self.similarity_fun)
                    results_data['budget'].append(b)
                    print('time consuming for attack:', datetime.now() - start)
                    pprint.pprint(results_data)
                    print(ranks[self.target_nodes.cpu().numpy()])

        return connectivity_scores_atk,results_data

    def evaluate(self,b,budget,results_data):
        K = budget
        if self.args.save_B_opt_loss:
            '''Get the optimal parameter B'''
            self.load_state_dict(torch.load(self.args.output_dir + 'model.pth'))
            print("Model's state_dict:")
            for param_tensor in self.state_dict():
                print(param_tensor, "\t", self.state_dict()[param_tensor].size())
            B = self.state_dict()[param_tensor]
        else:
            B = self.B.detach()
        '''Choose B according to topk'''
        topk_index = largest_indices(B.triu(), K)
        topk_index = [list(topk_index[0]), list(topk_index[1])]
        topk_B = torch.zeros(self.n, self.n).to(self.device)
        topk_B[topk_index[0], topk_index[1]] = B[torch.tensor(topk_index[0]).to(self.device),torch.tensor(topk_index[1]).to(self.device)]
        '''Show the results'''
        similarity_matrix = torch.abs(self.symmetric(topk_B) - self.similarity_matrix)
        with open(self.args.output_dir + f'B_{self.similarity_fun}_{b}.pth', 'wb') as f:
            pickle.dump([B,topk_B,self.target_nodes], f)
        print("min and max of attacked similarity_matrix:",similarity_matrix.min(),similarity_matrix.max())
        similarity_matrix = similarity_matrix.cpu().detach().numpy()
        similarity_matrix = (similarity_matrix > self.thre).astype(int) * similarity_matrix
        normalized_transition_matrix = np.divide(similarity_matrix, np.sum(similarity_matrix, axis=1))
        connectivity_scores_atk = PageRank_matrix(normalized_transition_matrix, c=0.15)
        targets_scores = [connectivity_scores_atk[t] for t in self.target_nodes]
        others_scores = [connectivity_scores_atk[l] for l in list(set(range(self.n)) - set(self.target_nodes))]
        print('***Attacked targets connectivity scores')
        print('minimum:', np.min(targets_scores))
        print('avarage:', np.mean(targets_scores))
        ranks = rankdata(connectivity_scores_atk)
        results_data['anomaly score'].append((1 - connectivity_scores_atk)[self.target_nodes.cpu().numpy()])
        results_data['ranking'].append(ranks[self.target_nodes.cpu().numpy()].mean() / self.args.data_size)
        results_data['detect top1%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.01))
        results_data['detect top5%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.05))
        results_data['detect top10%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.1))
        results_data['similarity'].append(self.similarity_fun)
        return connectivity_scores_atk,similarity_matrix,B, topk_B,results_data


    def baseline_attack(self,n_flips,results_data,labels):
        adj_matrix=(self.similarity_matrix.cpu().numpy() > 0).astype(int)
        candidates= generate_candidates_addition(adj_matrix,self.target_nodes.cpu().numpy())
        if self.args.attack_mode=="random":
            flips=baseline_random_top_flips(candidates, n_flips, self.args.random_seed)
        elif self.args.attack_mode=="degree":
            flips = baseline_degree_top_flips(adj_matrix, candidates, n_flips, False)
        elif self.args.attack_mode == "DeepWalk":
            index = np.array(random.sample(range(0, candidates.shape[0]), min(20000, len(candidates))))
            candidates = candidates[index]
            save_dir = f'{self.args.output_dir}/DeepWalk_data_{self.similarity_fun}.pkl'
            if os.path.isfile(save_dir)==False:
                adj_matrix[np.diag_indices_from(adj_matrix)] = 0.0
                np.isinf(adj_matrix).any()
                adj_matrix = sparse.csr_matrix(adj_matrix)
                flips = target_perturbation_top_flips(adj_matrix, candidates, n_flips, 32,self.target_nodes.cpu().numpy(),labels, save_dir=save_dir)
            else:
                f = open(save_dir, 'rb')
                candidate_margin_mean=pickle.load(f)[0]
                f.close()
                flips = candidates[candidate_margin_mean.argsort()[-n_flips:]]
        similarity_matrix = flip_candidates(self.similarity_matrix.cpu().numpy(), flips)
        normalized_transition_matrix = np.divide(similarity_matrix, np.sum(similarity_matrix, axis=1))
        connectivity_scores_atk = PageRank_matrix(normalized_transition_matrix, c=0.15)
        targets_scores = [connectivity_scores_atk[t] for t in self.target_nodes]
        others_scores = [connectivity_scores_atk[l] for l in list(set(range(self.n)) - set(self.target_nodes))]
        print('***Attacked targets connectivity scores')
        print('minimum:', np.min(targets_scores))
        print('avarage:', np.mean(targets_scores))
        ranks = rankdata(connectivity_scores_atk)
        results_data['anomaly score'].append((1 - connectivity_scores_atk)[self.target_nodes.cpu().numpy()])
        results_data['ranking'].append(ranks[self.target_nodes.cpu().numpy()].mean() / self.args.data_size)
        results_data['detect top1%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.01))
        results_data['detect top5%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.05))
        results_data['detect top10%'].append(1 - detected(ranks, self.target_nodes.cpu().numpy(), cutoff=0.1))
        results_data['similarity'].append(self.similarity_fun)
        return connectivity_scores_atk,results_data


