from networkx.algorithms import bipartite
import random
import networkx as nx
import numpy as np
import os
import heapq
import pickle
import torch
import torch.nn as nn
import torch.optim
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import spdiags,csr_matrix, find
from scipy.linalg import eig
from scipy.stats import rankdata
import torch.nn.functional as F
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from DeepWalkAttack.node_embedding_attack.perturbation_attack import *
from DeepWalkAttack.node_embedding_attack.utils import *
from utils import *


class RandomWork():
    '''Random walk based anomaly detection'''
    def __init__(self,G,k,n,c,args):
        self.G=G
        self.k=k
        self.n=n
        self.c=c
        self.device = args.device
        #self.adj_matrix=nx.adjacency_matrix(G).toarray()
        #self.M=self.adj_matrix[:self.k,self.k:]
        self.M,self.adj_matrix=self.get_M_and_adj_matrix(G,k,n)
        self.normalized_transition_matrix= np.divide(self.adj_matrix, np.sum(self.adj_matrix,axis=1))
        self.normalized_transition_matrix[np.isnan(self.normalized_transition_matrix)] = 0
        self.I = torch.eye(self.n + self.k).to(self.device)
        #self.normalized_transition_matrix[self.normalized_transition_matrix != self.normalized_transition_matrix] = 0
    def get_M_and_adj_matrix(self,G,k,n):
        edges_list = G.edges
        M = np.zeros((k, n))
        adj_matrix = np.zeros((k + n, k + n))
        for e in edges_list:
            M[e[0], e[1] - self.k] = 1
            adj_matrix[e[0], e[1]] = 1
        adj_matrix = adj_matrix + adj_matrix.T
        return M, adj_matrix

    def PageRank(self,P, c, root):
        """
        Personal Rank in matrix formation
        :param P: transfer probability matrix
        """
        result = []
        n = len(P)
        e = np.zeros(n)
        e[root] = 1
        s = np.zeros(n)
        s[root] = 1
        P=csr_matrix(P)
        #s=csr_matrix(s)
        #e = csr_matrix(e)
        while np.sum(abs(s - ((1 - c) * (P*s)  + c*e))) > 0.001:
            s = (1 - c) * (P*s) + c*e
        # while np.sum(abs(s - ((1 - c) * np.matmul(P, s) + c * e))) > 0.001:
        #     s = (1 - c) * np.matmul(P, s) + c * e
        return s

    def PageRank_matrix_ori(self,P, c):
        """
        Personal Rank in matrix formation
        :param P: transfer probability matrix
        """
        n = len(P)
        e = np.eye(self.k + self.n)[:,:self.k]
        s = np.eye(self.k + self.n)[:,:self.k]
        P=csr_matrix(P)
        e = csr_matrix(e)
        s = csr_matrix(s)

        while np.sum(abs(s - ((1 - self.c)* P@ s + self.c* e))) > 0.0001:
            s= (1 - self.c)*P@s + self.c* e

        return s.toarray().transpose()

    def PageRank_matrix(self,P, c):
        """
        Personal Rank in matrix formation
        :param P: transfer probability matrix
        """
        e = torch.from_numpy(np.eye(self.k + self.n)[:,:self.k]).to(self.device)
        s = torch.from_numpy( np.eye(self.k + self.n)[:,:self.k]).to(self.device)
        while torch.sum(torch.abs(s - ((1 - self.c)* torch.matmul(P,s)  + self.c* e))) > 0.0001:
            s= (1 - self.c) * torch.matmul(P,s)  + self.c* e
        return s.cpu().numpy().transpose()

    def get_graph(self):
        print('***edge matrix (M) ***')
        #print(self.M[0:10,0:10])
        print('shape:',self.M.shape)
        print('***adj_matrix***')
        #print(self.adj_matrix[0:10:,0:10])
        print('shape:',self.adj_matrix.shape)
        print('***normalized_transition_matrix***')
        #print(self.normalized_transition_matrix[0:10:,0:10])
        print('shape:',self.normalized_transition_matrix.shape)
        return self.M,self.adj_matrix,self.normalized_transition_matrix

    def get_all_similarity_scores(self):
        All_similarity_scores=[]
        P=self.normalized_transition_matrix
        P = torch.DoubleTensor(P).to(self.device)
        print('***Calculating similarity scores among U***')
        #for u in tqdm(range(self.k)):
        #    All_similarity_scores.append(self.PageRank(P, self.c, u))##好像可以全部一起算，一起乘，不用分开向量
        #All_similarity_scores=self.PageRank_matrix(P, self.c)
        All_similarity_scores_cls = self.c * torch.inverse(self.I - (1 - self.c) * P)[:,:self.k].transpose(0,1)
        return All_similarity_scores_cls.cpu().numpy()

    def get_mean_scores(self,M,All_similarity_scores,nodes):
        All_similarity_scores = torch.as_tensor(np.array(All_similarity_scores))
        All_similarity_scores = All_similarity_scores.type(torch.FloatTensor)
        All_similarity_scores = All_similarity_scores[:, :self.k]
        diag = torch.diag(All_similarity_scores)
        a_diag = torch.diag_embed(diag)
        Similarity = All_similarity_scores - a_diag
        Similarity.size()
        M=torch.as_tensor(M).type(torch.FloatTensor)
        MM = M.transpose(0,1).unsqueeze(dim=1)
        MM = MM[nodes]
        temp = torch.matmul(Similarity, MM.transpose(1, 2))
        sum_score = torch.matmul(temp.transpose(1, 2), MM.transpose(1, 2))
        neighbers = torch.sum(MM, dim=2).squeeze()
        neighbers_number=(neighbers*neighbers-neighbers)/10
        All_mean_S = torch.div(sum_score.squeeze(), neighbers_number)
        All_mean_S[torch.isnan(All_mean_S)] = 0
        All_mean_S=np.asarray(All_mean_S)
        return All_mean_S


class Attacker_optimization(nn.Module):
    '''Attacker optimization: get the best solution to attack'''
    def __init__(self, G, k, n, c,All_similarity_scores,args):
        super(Attacker_optimization, self).__init__()
        self.G=G
        self.k=k
        self.n=n
        self.c=c
        self.device = args.device
        M, adj_matrix = self.get_M_and_adj_matrix(G,k,n)
        self.M, self.adj_matrix =torch.as_tensor(M).type(torch.FloatTensor).to(self.device), torch.as_tensor(adj_matrix).type(torch.FloatTensor).to(self.device)
        self.normalized_transition_matrix = torch.div(self.adj_matrix, self.adj_matrix.sum(1))
        self.normalized_transition_matrix[torch.isnan(self.normalized_transition_matrix)] = 0
        self.B = nn.Parameter(torch.Tensor(self.k, self.n).to(self.device))
        nn.init.xavier_normal_(self.B)
        self.B.requires_grad=True
        All_similarity_scores = torch.as_tensor(np.array(All_similarity_scores))
        All_similarity_scores = All_similarity_scores.type(torch.FloatTensor)
        self.All_similarity_scores=All_similarity_scores.to(self.device)
        self.I=torch.eye(self.n+self.k).to(self.device)

    def get_M_and_adj_matrix(self,G,k,n):
        edges_list = G.edges
        M = np.zeros((k, n))
        adj_matrix = np.zeros((k + n, k + n))
        for e in edges_list:
            M[e[0], e[1] - self.k] = 1
            adj_matrix[e[0], e[1]] = 1
        adj_matrix = adj_matrix + adj_matrix.T
        return M, adj_matrix

    def update_all_similarity_scores(self):
        All_s=self.All_similarity_scores.transpose(0,1)
        P = torch.zeros(self.k + self.n, self.k + self.n).to(self.device)
        P[:self.k, self.k:] = torch.abs(self.B - self.M)
        P[self.k:, :self.k] = (torch.abs(self.B - self.M)).transpose(0, 1)
        # to prevent nan
        normalized_P = torch.div(P, P.sum(1))
        normalized_P[torch.isnan(normalized_P)] = 0
        self.All_similarity_scores=((1 - self.c) * torch.matmul(normalized_P, All_s)+self.c * torch.eye(self.k+self.n)[:,:self.k].to(self.device)).transpose(0, 1)
        return self.All_similarity_scores,normalized_P

    def get_mean_scores(self,M,All_similarity_scores,nodes):
        All_similarity_scores = All_similarity_scores[:, :self.k]
        diag = torch.diag(All_similarity_scores)
        a_diag = torch.diag_embed(diag)
        Similarity = All_similarity_scores - a_diag
        Similarity.size()
        M=torch.as_tensor(M).type(torch.FloatTensor).to(self.device)
        MM = M.transpose(0,1).unsqueeze(dim=1)
        MM = MM[nodes]
        temp = torch.matmul(Similarity, MM.transpose(1, 2))
        sum_score = torch.matmul(temp.transpose(1, 2), MM.transpose(1, 2))
        neighbers=torch.sum(MM, dim=2).squeeze()
        neighbers_number=(neighbers*neighbers-neighbers)/100
        All_mean_S = torch.div(sum_score.squeeze(), neighbers_number)
        All_mean_S[torch.isnan(All_mean_S)] = 0
        return All_mean_S

    def updata_M(self,M):
        '''This only used in GradMax, which only modify one edge at once'''
        self.M=M
        adj_matrix = torch.zeros(self.k + self.n, self.k + self.n)
        adj_matrix[:self.k, self.k:] =  self.M
        adj_matrix[self.k:, :self.k] = self.M.transpose(0, 1)
        self.adj_matrix=adj_matrix
        nn.init.zeros_(self.B)
        self.normalized_transition_matrix = torch.div(self.adj_matrix, self.adj_matrix.sum(1))
        self.normalized_transition_matrix[torch.isnan(self.normalized_transition_matrix)] = 0
        return None

    def atk_loss(self,targets_scores,allnode_scores):#attack loss function
        # input is the vector of connectivity scores
        ret=-torch.sum(targets_scores)/torch.sum(allnode_scores)*1000
        ret[torch.isnan(ret)]=0
        return ret

    def forward(self,targets):
        All_similarity_scores,_=self.update_all_similarity_scores()
        with torch.autograd.set_detect_anomaly(True):
            M=torch.abs(self.B - self.M)
            allnode_scores=self.get_mean_scores(M, All_similarity_scores, range(self.n))
            targets_scores = allnode_scores[targets]
        return targets_scores,allnode_scores

    def cf_forward(self,targets):
        M = torch.abs(self.B - self.M)
        P = torch.zeros(self.k + self.n, self.k + self.n).to(self.device)
        P[:self.k, self.k:] = M
        P[self.k:, :self.k] = M.transpose(0, 1)
        # to prevent nan
        normalized_P = torch.div(P, P.sum(1))
        normalized_P[torch.isnan(normalized_P)] = 0
        All_similarity_scores = self.c * torch.inverse(self.I -(1-self.c)*normalized_P)[:,:self.k].transpose(0,1)
        #with torch.autograd.set_detect_anomaly(True):
        allnode_scores=self.get_mean_scores(M, All_similarity_scores, range(self.n))
        targets_scores = allnode_scores[targets]
        return targets_scores, allnode_scores


class Get_attacked_detect_score(RandomWork):
    '''get detection scores after attacked'''
    def __init__(self, G,k,n,c,topk_B,targeted_nodes,args):
        super().__init__(G,k,n,c,args)
        self.topk_B=topk_B
        self.targeted_nodes=targeted_nodes
        self.adj_matrix,self.M = self.get_attacked_A_and_M()
        print("min and max of attacked adj_matrix:", self.adj_matrix.min(), self.adj_matrix.max())
        self.normalized_transition_matrix=np.divide(self.adj_matrix, np.sum(self.adj_matrix,axis=1))
        #self.normalized_transition_matrix[torch.isnan(self.normalized_transition_matrix)] = 0
        #np.isnan(self.normalized_transition_matrix).sum()
        self.normalized_transition_matrix[self.normalized_transition_matrix != self.normalized_transition_matrix] = 0

    def get_attacked_A_and_M(self):
        # BB=(O B; B' O);
        BB = np.zeros((self.k + self.n, self.k + self.n))
        BB[:self.k, self.k:] = self.topk_B
        BB[self.k:, :self.k] = self.topk_B.transpose()
        attacked_adj_matrix = np.abs(self.adj_matrix - BB)
        attacked_M = np.abs(self.M - self.topk_B)
        return attacked_adj_matrix,attacked_M

    def get_targets_scores(self):
        All_similarity_scores=self.get_all_similarity_scores()
        #targets_scores=self.get_mean_scores(self.M,All_similarity_scores,self.targeted_nodes)
        allnode_scores = self.get_mean_scores(self.M, All_similarity_scores, range(self.n))
        targets_scores = np.array([allnode_scores[t] for t in self.targeted_nodes])
        others_scores = np.array([allnode_scores[l] for l in list(set(range(self.n))-set(self.targeted_nodes))])
        return targets_scores,others_scores,allnode_scores

'''Attacker functions'''


def attack(b,budget,target_nodes,G,k,n,c,All_similarity_scores,result_data,args):
    K = budget
    start = datetime.now()
    attacker = Attacker_optimization(G, k, n, c, All_similarity_scores,args)
    constraints = ConstraintProjector(K,args.scaling)
    target_nodes = torch.as_tensor(target_nodes)
    if args.opt=='SGD':
        opt = torch.optim.SGD(attacker.parameters(), lr=args.lr)  # weight_decay=0.0001
    elif args.opt=='Adam':
        opt = torch.optim.Adam(attacker.parameters(), lr=args.lr,weight_decay=args.lamda)

    min_loss = 1000000
    for i in range(args.attack_epoch):
        opt.zero_grad()
        if args.attack_mode=="alternative":
            targets_scores, allnode_scores = attacker.forward(target_nodes)
        elif args.attack_mode=="closed-form":
            targets_scores, allnode_scores = attacker.cf_forward(target_nodes)
        loss1 = attacker.atk_loss(targets_scores, allnode_scores)
        loss = loss1 + args.lamda * constraint_loss(attacker.B, K)
        print(i, 'loss:', loss1.item(), 'sum(B)-K:', (attacker.B.sum() - K).item())
        loss.backward(retain_graph=True)
        if args.attack_mode == "closed-form":
            torch.nn.utils.clip_grad_norm_(attacker.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            attacker.apply(constraints)
        if loss1 < min_loss and i > 1:
            min_loss = loss1
            print("save model")
            torch.save(attacker.state_dict(), args.output_dir + 'model.pth')
        if np.isnan(loss1.item()):
            break
    print('time consuming for attack:', datetime.now() - start)

    '''Get the optimal parameter B'''
    attacker.load_state_dict(torch.load(args.output_dir + 'model.pth'))
    print("Model's state_dict:")
    for param_tensor in attacker.state_dict():
        print(param_tensor, "\t", attacker.state_dict()[param_tensor].size())
    B = attacker.state_dict()[param_tensor].cpu()
    # with open(args.output_dir + 'B.pkl', 'wb') as f:
    #     pickle.dump(B, f)

    '''Binarize B according to topk'''
    B_binary = (B > 0.5)
    B_binary = np.asarray(B_binary)
    print('B_binary>0.5的个数, max:', np.sum(B_binary != 0), B.max())
    topk_index = largest_indices(B, K)
    topk_index = [list(topk_index[0]), list((topk_index[1]))]
    topk_B = np.zeros((k, n))
    for i in range(len(topk_index[0])):
        topk_B[topk_index[0][i], topk_index[1][i]] = 1

    '''Show the results'''
    attacked_detect = Get_attacked_detect_score(G, k, n, c, topk_B, np.asarray(target_nodes),args)
    targets_scores, others_scores, allnode_scores = attacked_detect.get_targets_scores()
    print('***Attacked targets mean similarity scores')
    print('minimum:', np.min(targets_scores))
    print('avarage:', np.mean(targets_scores))

    ranks = rankdata(allnode_scores)
    result_data['maximum score'].append((1 - targets_scores).max())
    result_data['avarage score'].append((1 - targets_scores).mean())
    result_data['ranking'].append(ranks[target_nodes].mean() / n)
    result_data['detected_top1%'].append(detected(ranks, target_nodes, cutoff=0.01))
    result_data['detected_top5%'].append(detected(ranks, target_nodes, cutoff=0.05))
    result_data['detected_top10%'].append(detected(ranks, target_nodes, cutoff=0.1))
    result_data['budget'].append(b)
    return targets_scores, others_scores,allnode_scores,result_data

class ConstraintProjector(object):
    '''projected gradient descend'''

    def __init__(self, K, scaling=False):
        self.K = K
        self.scaling = scaling

    def __call__(self, module):
        if hasattr(module, 'B'):
            print("apply Constraint projection")
            w = module.B.data
            w = w.clamp(0, 1)  # projected B to [0,1]
            if self.scaling:
                total = w.sum()
                if total > 1000:
                    w = w * (self.K*10 / total)  # scale the B as: sum(B)=K
                    w[torch.isnan(w)] =0
            # w[torch.isnan(w)] =0
            module.B.data = w


def constraint_loss(B, K):
    '''loss1 is the original loss of model'''
    sumB = B.sum()
    loss = torch.abs(sumB - K).sum()  # + lamda * torch.abs(sumB - K)#add constraint to puch sum(B) near to K
    return loss


def evaluate_baseline(adj_matrix,M,c,detector,all_v_nodes,n_flips,result_data,target_nodes,pseudo_labels,args):
    k,n=M.shape
    candidates= generate_candidates_addition(M,target_nodes)
    if args.attack_mode=="random":
        flips=baseline_random_top_flips(candidates, n_flips, args.random_seed)
    elif args.attack_mode=="degree":
        flips = baseline_degree_top_flips(adj_matrix, candidates, n_flips, False)
    elif args.attack_mode=="DeepWalk":
        candidates_remove = generate_candidates_removal(sparse.csr_matrix(adj_matrix))
        candidates=np.concatenate((candidates,candidates_remove),axis=0)
        print("number of candidate flips:",min(20000, len(candidates)))
        index = np.array(random.sample(range(0, candidates.shape[0]), min(20000, len(candidates))))
        candidates = candidates[index]
        save_dir = f'{args.output_dir}/DeepWalk_data.pkl'
        if os.path.isfile(save_dir) == False:
            adj_matrix[np.diag_indices_from(adj_matrix)] = 0.0
            np.isinf(adj_matrix).any()
            adj_matrix = sparse.csr_matrix(adj_matrix)
            flips = target_perturbation_top_flips(adj_matrix,k, candidates, n_flips, 32, target_nodes,pseudo_labels,save_dir=save_dir)
            adj_matrix=adj_matrix.toarray()
        else:
            f = open(save_dir, 'rb')
            sum_delta_eigvecs = pickle.load(f)[0]
            f.close()
            flips = candidates[sum_delta_eigvecs.argsort()[-n_flips:]]
            # flips = candidates[sum_delta_eigvecs.argsort()[:n_flips]]
    adj_matrix_atk = np.array(flip_candidates(adj_matrix, flips))
    normalized_transition_matrix = np.divide(adj_matrix_atk, np.sum(adj_matrix_atk, axis=1))
    normalized_transition_matrix[np.isnan(normalized_transition_matrix)] = 0
    I = torch.eye(adj_matrix_atk.shape[0]).to(args.device)
    P = torch.DoubleTensor(normalized_transition_matrix).to(args.device)
    All_similarity_scores_cls = c * torch.inverse(I - (1 - c) * P)[:, :k].transpose(0, 1)
    All_mean_Scores = detector.get_mean_scores(M, All_similarity_scores_cls.cpu(), all_v_nodes)
    ranks = rankdata(All_mean_Scores)
    result_data['maximum score'].append((1 - All_mean_Scores)[target_nodes].max())
    result_data['avarage score'].append((1 - All_mean_Scores)[target_nodes].mean())
    result_data['ranking'].append(ranks[target_nodes].mean() / n)
    result_data['detected_top1%'].append(detected(ranks, target_nodes, cutoff=0.01))
    result_data['detected_top5%'].append(detected(ranks, target_nodes, cutoff=0.05))
    result_data['detected_top10%'].append(detected(ranks, target_nodes, cutoff=0.1))
    return All_mean_Scores,result_data
