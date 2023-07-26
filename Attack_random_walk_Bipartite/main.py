from networkx.algorithms import bipartite
import random
import networkx as nx
import numpy as np
import os
import pprint
import heapq
import argparse
import json
import pickle
import shutil
import torch
import torch.nn as nn
import torch.optim
from scipy import sparse
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, csr_matrix, find
from DeepWalkAttack.node_embedding_attack.perturbation_attack import *
from sklearn.metrics import roc_curve, auc
from scipy.linalg import eig
import torch.nn.functional as F
from datetime import datetime
from models import *
from utils import *
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
plt.style.use('classic')
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['legend.title_fontsize'] =BIGGER_SIZE
# Model Settings==============================================================
parser = argparse.ArgumentParser(description='attack RW model')
parser.add_argument('-gpuID', type=int, default=0)
parser.add_argument('-random_seed', type=int, default=2021)
parser.add_argument('-lr', type=float, default=1.0, metavar='LR', help='learning rate')
parser.add_argument('-lamda', type=float, default=1e-6)
parser.add_argument('-data_size', type=int, default=10000)
parser.add_argument('-degree_threshold', type=int, default=5, help="filter the low degree nodes in the data")
# Node Injection Attack setting----------------------
parser.add_argument('-target_node_num', type=int, default=5)
parser.add_argument('-random_node_num', type=int, default=100,help='random sample the anomaly nodes from top anomaly nodes')
parser.add_argument('-attack_epoch', type=int, default=60, metavar='N', help='training epoch')
parser.add_argument('-increasing_budget', action='store_true', default=True,
                    help="increasing attack edge budget proportion if ture; false:fix the proportion")
parser.add_argument('-fixed_budget', type=float, default=1.0, help="attack budget per target node: 1.0 * average degree * target nodes number")
parser.add_argument('-budget_mode', type=str, default='node_degree', choices=['totoal_edges', 'node_degree'],
                    help="totoal_edges: total budget = budget * total edges; node_degree : total budget= budget * target nodes * average degree of target node")
parser.add_argument('-scaling', action='store_true', default=False, help="scaling the parameters(B) while optimization")
parser.add_argument('-attack_mode', type=str, default='DeepWalk',
                    choices=['alternative', 'closed-form','random','degree','DeepWalk'])
parser.add_argument('-opt', type=str, default='SGD',
                    choices=['SGD', 'Adam'])
parser.add_argument('-tag', type=str, default='')
# dir setting----------------------------------------
parser.add_argument('-dataset', type=str, default='AuthorPapers',
                    choices=['RandomBipartite_ER', 'RandomBipartite_BA', 'AuthorPapers','Magzine'])
parser.add_argument('-output_dir', type=str, default='')
args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device(f'cuda:{args.gpuID}')
    print(f"---using GPU---cuda:{args.gpuID}----")
else:
    print("---using CPU---")
    args.device = torch.device("cpu")

if args.dataset == "AuthorPapers":
    args.data_size = 10000
    args.degree_threshold = 5
    bins_t = 4
if args.dataset == "AmazonReviews":
    args.data_size = 450000
    args.degree_threshold = 10
if args.dataset == "Magzine":
    args.data_size = 100000
    args.degree_threshold = 3
    bins_t = 15

if args.attack_mode=="closed-form" and args.dataset == "AuthorPapers":
    args.lamda=0.001
    args.lr=0.01
    args.scaling=False
    args.opt='Adam'
    # args.lamda=1e-6
    # args.lr=1.0
    # args.scaling=True
    # args.opt='SGD'

if args.attack_mode=="closed-form" and args.dataset == "Magzine":
    args.lamda=0.0001
    args.lr=0.1
    args.scaling=True
    args.opt='SGD'

if args.increasing_budget == False:
    args.output_dir = f'./results_{args.dataset}/{args.attack_mode}/e{args.attack_epoch}_lamda{args.lamda}_lr{args.lr}_scl{args.scaling}/b{args.budget_mode}_{args.random_seed}/'
else:
    if args.budget_mode == "totoal_edges":
        budget = [round(b, 2) for b in np.arange(0.01, 0.06, 0.01)]  # budget * total edges = total budget
        if args.dataset == "Magzine":
            budget = [round(b, 3) for b in np.arange(0.005, 0.026, 0.005)]  # budget * total edges = total budget
    else:
        budget = [round(b, 1) for b in
                  np.arange(0.2, 1.1, 0.2)]  # budget * average degree * target nodes number = total budget
    print(budget)
    args.output_dir = f'./results_{args.dataset}/{args.attack_mode}/e{args.attack_epoch}_lamda{args.lamda}_lr{args.lr}_scl{args.scaling}/b{args.budget_mode}_{args.random_seed}/'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
print("output dir:", args.output_dir)
init_random_seed(args.random_seed)
pprint.pprint(vars(args), width=1)
# ============================================================================

if False: # save the results to pkl file to reduce recalculation.
    # G is the networkx graph;k,n is the number of top nodes and bottom nodes.
    G, k, n = Dataloader(datasets=args.dataset, number_records=args.data_size)

    # A=nx.adjacency_matrix(G).toarray()
    # visualize_matrix(A[0:4000:,-4000:])

    '''remove bottom node with less than 2 degree and
    find the maximum connected components of G and remap the node id'''
    G, k, n = remove_low_degree(G, k, threshold=args.degree_threshold, iteration=10)

    largest_components = max(nx.connected_components(G), key=len)
    print(largest_components)
    print('number of nodes to keep:', len(largest_components))
    G, k, n = reconstruct_graph(G, keep_nodes=largest_components)

    '''visualization'''
    # visualization_graph_flag = False
    # visualization_graph(G, visualization_graph_flag, bipartite_layout=True)

    '''Injection of anomaly nodes'''
    k0 = int(k * 0.1)
    n0 = int(n * 0.1)
    print("injected malicious node:", n0)
    G, M2 = injectCliqueCamo(G, k, n, k0, n0, 0.05, type=1)
    # visualize_matrix(M2)
    labels = [0] * n
    labels.extend([1] * n0)
    labels=np.array(labels)
    '''Calculate all nodes and get high malicious nodes as targets'''
    c = 0.15
    n = n + n0

    detector = RandomWork(G, k, n, c,args)
    M, adj_matrix, normalized_transition_matrix = detector.get_graph()
    # visualize_matrix(M)
    All_similarity_scores = detector.get_all_similarity_scores()
    all_v_nodes = list(range(0, n))  # this index is based on M (k*n)
    All_mean_Scores = detector.get_mean_scores(M, All_similarity_scores, all_v_nodes)
    # visualize_anomaly_scores(G,k, n, n0, bins_t, labels,All_mean_Scores, args)

    '''random sample the anomaly nodes from top anomaly scores:'''
    sorted_index = np.argsort(All_mean_Scores)
    idx = [ind for i, ind in enumerate(sorted_index) if labels[ind] == 1][0:args.random_node_num]
    random.seed(args.random_seed)
    target_nodes = np.array(random.sample(idx, args.target_node_num))
    pseudo_labels = np.zeros(len(All_mean_Scores))
    for ind in sorted_index[:int(len(All_mean_Scores) * 0.1)]:
        pseudo_labels[ind]=1

    average_degree = np.sum(M > 0, axis=0)[target_nodes].mean()
    total_edges = (M > 0).sum()
    other_nodes = np.array(list(set(range(n)) - set(target_nodes)))
    # -----------------------------------------------------

    with open(args.output_dir + 'Detection_info.pkl', 'wb') as f:
        pickle.dump([G, k, n,M,adj_matrix,labels,pseudo_labels,All_similarity_scores, All_mean_Scores, target_nodes,average_degree,total_edges], f)
else:
    c = 0.15
    with open(args.output_dir + 'Detection_info.pkl', 'rb') as f:
        G, k, n,M,adj_matrix,labels,pseudo_labels,All_similarity_scores, All_mean_Scores, target_nodes, average_degree,total_edges=pickle.load(f)
    detector = RandomWork(G, k, n, c,args)
    all_v_nodes = list(range(0, n))
result_data = {'maximum score': [], 'avarage score': [], 'ranking': [], 'detected_top1%':[], 'detected_top5%':[], 'detected_top10%':[],
            'budget': []}

#detection performance:AUC curve
# draw_roc(labels, 1-All_mean_Scores, title=None, savedir=f'{args.output_dir}BiGraphROC.pdf')

ranks = rankdata(All_mean_Scores)
print('***Original targets connectivity scores')
result_data['maximum score'].append((1 - All_mean_Scores)[target_nodes].max())
result_data['avarage score'].append((1 - All_mean_Scores)[target_nodes].mean())
result_data['ranking'].append(ranks[target_nodes].mean() / n)
result_data['detected_top1%'].append(detected(ranks, target_nodes, cutoff=0.01))
result_data['detected_top5%'].append(detected(ranks, target_nodes, cutoff=0.05))
result_data['detected_top10%'].append(detected(ranks, target_nodes, cutoff=0.1))
result_data['budget'].append(0)
pprint.pprint(result_data)

# total_edges = np.sum((M > 0) != 0)
'''Attack main function from here'''
if args.increasing_budget == True:
    for b in budget:
        if args.budget_mode == "totoal_edges":
            total_budget=int(total_edges*b)
        else:
            total_budget = int(average_degree * b * args.target_node_num)
        print(f"budget: {b}, total budget: {total_budget}")

        ## baselines:
        if args.attack_mode in ['random' ,'degree','DeepWalk']:
            All_mean_Scores,result_data = evaluate_baseline(adj_matrix,M,c,detector,all_v_nodes,total_budget,result_data,target_nodes,pseudo_labels,args)
            if args.increasing_budget == True:
                result_data['budget'].append(b)
            pprint.pprint(result_data)
            print(ranks[target_nodes])
            continue

        ## our methods:
        targets_scores, others_scores, allnode_scores,result_data = attack(b,total_budget, target_nodes,G,k,n,c,All_similarity_scores, result_data,args)

        plt.figure(constrained_layout=True)
        if args.budget_mode == "totoal_edges":
            plt.title(f'Distribution of attacked anomaly scores \n with budget: {b*100}%')
        else:
            plt.title(f'Distribution of attacked anomaly scores \n with budget: {b}')
        plt.hist(1 - others_scores,bins=50, facecolor="darkorange", edgecolor="black", alpha=0.3)
        plt.hist(1 - targets_scores,bins=bins_t, facecolor="red", edgecolor="black", alpha=0.4)
        target = "red"
        other = "darkorange"
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k", alpha=0.5) for c in [target, other]]
        labels = ["target nodes", "other nodes"]
        plt.legend(handles, labels)
        plt.ylabel("frequency")
        plt.xlabel("anomaly scores")
        plt.legend(handles=handles, labels=labels, loc='upper left')
        plt.savefig(args.output_dir + f'AttackDistribution_budget{b}.pdf', dpi=300)
        plt.show()
        plt.clf()
else:
    targets_scores, others_scores = attack(b,args.fixed_budget, target_nodes,G,k,n,c,All_similarity_scores,result_data, args)

pprint.pprint(result_data)


'''visulization'''
if args.increasing_budget==True:
    draw_figures(result_data, args, combine=True)

'''Record the results'''
if os.path.exists(f'./results_{args.dataset}/result_summary.txt') == False:
    with open(f'./results_{args.dataset}/result_summary.txt', 'w') as f:
        f.write('result record\n')
        if args.increasing_budget == True:
            line = f"{args.attack_mode}:{args.budget_mode}_lr{args.lr}_epoch{args.attack_epoch}_lamda{args.lamda}_scal{args.scaling}_{args.opt}_{args.random_seed}: \n {result_data['ranking']}\n {result_data['detected_top5%']}"
        else:
            line = f"{args.attack_mode}:{args.budget_mode}_lr{args.lr}_epoch{args.attack_epoch}_lamda{args.lamda}_scal{args.scaling}_{args.opt}_b{args.fixed_budget}_{args.random_seed}: \n {result_data['ranking']}\n {result_data['detected_top5%']}"
        f.write(line)
        f.write('\n')
else:
    with open(f'./results_{args.dataset}/result_summary.txt', 'a') as f:
        if args.tag!='':
            f.write(args.tag)
            f.write('\n')
        if args.increasing_budget == True:
            line = f"{args.attack_mode}:{args.budget_mode}_lr{args.lr}_epoch{args.attack_epoch}_lamda{args.lamda}_scal{args.scaling}_{args.opt}_{args.random_seed}: \n {result_data['ranking']}\n {result_data['detected_top5%']}"
        else:
            line = f"{args.attack_mode}:{args.budget_mode}_lr{args.lr}_epoch{args.attack_epoch}_lamda{args.lamda}_scal{args.scaling}_{args.opt}_b{args.fixed_budget}_{args.random_seed}: \n {result_data['ranking']}\n {result_data['detected_top5%']}"
        f.write(line)
        f.write('\n')
