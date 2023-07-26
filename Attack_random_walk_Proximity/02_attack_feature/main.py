# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import random
import pickle
import torch
import seaborn as sns;
sns.set()
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc, precision_score
import torch.nn.functional as F
import argparse
import pprint
from utils import *
from model import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
# Model Settings==============================================================
parser = argparse.ArgumentParser(description='attack RW model')
parser.add_argument('-gpuID', type=int, default=0)
parser.add_argument('-random_seed', type=int, default=2024)
parser.add_argument('-data_size', type=int, default=10000)
parser.add_argument('-similarity_threshold', type=float, default=0.8, help="threshold for constructing proximity graph")
# Attack setting----------------------
parser.add_argument('-lr', type=float, default=1.0, help='learning rate')
parser.add_argument('-lamda', type=float, default=0.0)
parser.add_argument('-attack_epoch', type=int, default=500, help='training epoch')
parser.add_argument('-target_node_num', type=int, default=20)
parser.add_argument('-random_node_num', type=int, default=100,help='random sample the anomaly nodes from top anomaly nodes')
parser.add_argument('-increasing_budget', action='store_true', default=True,
                    help="increasing attack budget if ture; false:fix the budget")
parser.add_argument('-fixed_budget', type=float, default=10.0, help="attack budget")
parser.add_argument('-budget_mode', type=str, default='node_degree', choices=['totoal_edges', 'node_degree','attack_number'],
                    help="totoal_edges: total budget = budget * total edges; node_degree : total budget= budget * target nodes * average degree of target node; attack_number: number of attack ndoes")
parser.add_argument('-include_target', action='store_true', default=False, help="includes the target ndoes as attack nodes or not")
parser.add_argument('-save_opt', action='store_true', default=False,
                    help="save the parameters with lowest loss during training if Ture, else, use the One in final step")
parser.add_argument('-apply_constraint', action='store_true', default=True, help="apply features constraints if True")
parser.add_argument('-attack_mode', type=str, default='graph_guided_cf',
                    choices=['graph_guided', 'random', 'random_low_degree','random_high_degree','graph_guided_cf'],
                    help="graph_guided feature attack, and baseline(random) attack, run the graph_guided first")
parser.add_argument('-attack_loss', type=str, default='target_anomaly',
                    choices=['target_anomaly','target_anomaly_adaptive','attacked_graph'],
                    help="attacked_graph: make the features be close to the attacked graph, it must used with attack_mode:graph_guided and include_target:False")
parser.add_argument('-analyze_results', action='store_true', default=True, help="analyze the attack results")
# dir setting----------------------------------------
parser.add_argument('-dataset', type=str, default='KDD99', choices=['KDD99', 'Musk','Satelite','Satimage','Mnist'])
parser.add_argument('-output_dir', type=str, default='')
args = parser.parse_args()
# ============================================================================

if args.dataset == "KDD99":
    args.input_dir = "../DataSets/NetworkIntrusion/KDD99/"
    args.file_name = "kddcup.data_10_percent"
    args.data_size = 10000
    '''Preprocessing '''
    args.data_size = args.data_size + int(args.data_size * 0.01)
    features, labels = load_data(args.input_dir)
    labels = np.array([0 if l == "normal" else 1 for l in labels])
    total_edge_budget = [round(b, 4) for b in np.arange(0.0001, 0.0006, 0.0001)]
    args.graph_dir = f'../01_attack_graph/results_KDD99/alternative/e35_lamda0.0001_lr1.0_sFalse/target20_thre0.8_{args.random_seed}/'
elif args.dataset == "Musk":
    args.input_dir = "../DataSets/Musk"
    args.file_name = "musk.mat"
    args.target_node_num=5
    args.attack_epoch=200
    features, labels = load_FromMat(os.path.join(args.input_dir, args.file_name))
    args.data_size = features.shape[0]
    args.graph_dir = f'../01_attack_graph/results_Musk/e80_lamda0.0001_lr0.1_sFalse/target5_thre0.5_{args.random_seed}/'
    total_edge_budget = [round(b, 3) for b in np.arange(0.002, 0.012, 0.002)]
    args.similarity_threshold = 0.5
    args.budget_mode="attack_number"
elif args.dataset == "Mnist":
    args.input_dir = "../DataSets/Mnist"
    args.file_name = "mnist.mat"
    args.target_node_num=20
    features, labels = load_FromMat(os.path.join(args.input_dir, args.file_name))
    args.data_size = features.shape[0]
    args.graph_dir = f'../01_attack_graph/results_Mnist/alternative/e100_lamda0.0001_lr0.01_sFalse/target20_thre0.5_{args.random_seed}/'
    total_edge_budget = [round(b, 3) for b in np.arange(0.002, 0.012, 0.002)]
    args.similarity_threshold = 0.5
    args.apply_constraint=False
    #args.budget_mode="attack_number"

if args.attack_loss == "attacked_graph":
    if args.dataset=="KDD99":
        args.lr = 10.0
    args.attack_epoch = 1000
    assert(args.attack_mode in ['graph_guided'])
    args.include_target = False

if args.attack_loss in[ "random", "random_low_degree","random_high_degree",'graph_guided_cf']:
    args.attack_loss = "target_anomaly"

if torch.cuda.is_available():
    args.device = torch.device(f'cuda:{args.gpuID}')
    print(f"---using GPU---cuda:{args.gpuID}----")
else:
    print("---using CPU---")
    args.device = torch.device("cpu")

args.output_dir = f'./results_{args.dataset}/{args.attack_mode}/{args.attack_loss}/e{args.attack_epoch}_ld{args.lamda}_lr{args.lr}_c{args.apply_constraint}_b{args.budget_mode}_s{args.save_opt}/{args.random_seed}/'
if args.increasing_budget == False:
    budget = [args.fixed_budget]  # budget * average degree = total budget
    results_data = {'avarage score': [], 'ranking': [], 'detect top1%': [], 'detect top5%': [],
                    'detect 10%': [], 'similarity': []}
else:
    if args.budget_mode == "totoal_edges":
        budget = total_edge_budget  # budget * total edges = total budget
        decimal = 2  # for drawing figures
    elif args.budget_mode == 'node_degree':
        budget = [0.2,0.4,0.6,0.8,1.0]  # budget * average degree * target nodes number = total budget
        decimal = 1  # for drawing figures
    else:
        budget = [200,10,20,50,100,150,200]
        decimal=0
    results_data = {'anomaly score': [], 'ranking': [], 'detect top1%': [], 'detect top5%': [],
                    'detect top10%': [], 'similarity': [],
                    'budget': []}

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
print("output dir:", args.output_dir)
init_random_seed(args.random_seed)
pprint.pprint(vars(args), width=1)

original_feature = torch.FloatTensor(features).to(args.device)
orginal_labels = labels
for similarity_fun in ['correlation','cosine']:
    if similarity_fun == 'cosine':
        thre = args.similarity_threshold
    if similarity_fun == 'correlation':
        thre = args.similarity_threshold

    '''Evaluating anomaly scores '''
    similarity_matrix, _, connectivity_scores = get_connectivity_scores(original_feature,similarity_fun, thre)
    # draw_roc(orginal_labels, 1 - connectivity_scores, title=similarity_fun,
    #          savedir=f'{args.output_dir}ROC_{similarity_fun}.pdf')
    sorted_index = np.argsort(connectivity_scores, kind='stable')  # Ascending order
    # sort by connectivity scores
    features_1, connectivity_scores_1, similarity_matrix_1, labels_1 = sort_features_by_index(
        sorted_index, original_feature, connectivity_scores, similarity_matrix,
        orginal_labels)

    #random sample the anomaly nodes from top anomaly scores:
    idx=[i for i,l in enumerate(labels_1) if l==1][0:args.random_node_num]
    random.seed(args.random_seed)
    target_nodes=np.array(random.sample(idx, args.target_node_num))
    target_nodes = target_nodes.astype(int)
    nontarget_nodes = list(np.setdiff1d(np.array(range(0, args.data_size)), target_nodes))
    total_edges = (similarity_matrix_1 > 0).sum() / 2
    average_degree = np.sum(similarity_matrix_1 > 0, axis=0)[target_nodes].mean()  # average degree of target nodes
    # -----------------------------------------------------

    ranks = rankdata(connectivity_scores_1)
    results_data['anomaly score'].append(1 - connectivity_scores_1[target_nodes])
    results_data['ranking'].append(ranks[target_nodes].mean() / args.data_size)
    results_data['detect top1%'].append(1 - detected(ranks, target_nodes, cutoff=0.01))
    results_data['detect top5%'].append(1 - detected(ranks, target_nodes, cutoff=0.05))
    results_data['detect top10%'].append(1 - detected(ranks, target_nodes, cutoff=0.1))
    results_data['similarity'].append(similarity_fun)
    if args.increasing_budget == False:
        results_data['epochs'].append(0)
    else:
        results_data['budget'].append(0)
    pprint.pprint(results_data)
    print(ranks[target_nodes])

    '''for results analysis'''
    if args.analyze_results:
        analysis_data = {'edge_modified': [], 'control_degrees': [], 'budget': []}

    for b in budget:
        controlled_nodes,topk_B = choose_control_nodes(args,target_nodes,nontarget_nodes,similarity_fun,b,similarity_matrix_1)
        print("number of controlled nodes:",len(controlled_nodes))
        other_nodes = np.setdiff1d(np.array(range(0, args.data_size)), controlled_nodes)
        sorted_index = np.append(controlled_nodes, other_nodes).astype('int')
        features_2, connectivity_scores_2, similarity_matrix_2, labels_2 = sort_features_by_index(
            sorted_index, features_1, connectivity_scores_1, similarity_matrix_1, labels_1)
        map = dict(zip(sorted_index, range(0, args.data_size)))
        target_nodes_2 = np.array([map[n] for n in target_nodes])
        controlled_nodes = np.array([map[n] for n in controlled_nodes])
        if args.attack_loss == "attacked_graph":
            topk_B = topk_B[sorted_index, :]
            topk_B = topk_B[:, sorted_index]
            edges = torch.tensor(topk_B.cpu().numpy()[:,controlled_nodes].nonzero()).to(args.device)  #the edges between target node - control nodes or control node - control node
            print("attacked edges number:",len(edges[0]))
            target_similarity = torch.abs(topk_B[:,controlled_nodes][edges[0],edges[1]]- torch.tensor(similarity_matrix_2).to(args.device)[:,controlled_nodes][edges[0],edges[1]])
            attacked_graph_info=[edges,target_similarity,edges[0].shape[0]]
        else:
            attacked_graph_info=None
        attacker = Attacker_optimization(b, features_2, connectivity_scores_2, 0.15, target_nodes_2, similarity_fun,
                                         controlled_nodes, thre, args, atk_graph=attacked_graph_info).to(args.device)
        connectivity_scores_atk, similarity_matrix_atk, features_atk, results_data = attacker.attack(results_data)
        with open(args.output_dir + f'feature_atk_{similarity_fun}_{b}.pkl', 'wb') as f:
            pickle.dump([target_nodes_2,features_2,features_atk], f)
        with open(args.output_dir + f'labels_{similarity_fun}_{b}.pkl', 'wb') as f:
            pickle.dump([labels_2], f)
        if args.increasing_budget == True:
            results_data['budget'].append(b)
        pprint.pprint(results_data)

        '''analyze the attack results'''
        if args.analyze_results:
            analysis_data = analyze_results(b, analysis_data, controlled_nodes,
                                        similarity_matrix_2, similarity_matrix_atk, target_nodes_2, total_edges)#average_degree*args.target_node_num
    if args.analyze_results:
        draw_analysis_figures(similarity_fun,analysis_data, 2, args, average_degree*args.target_node_num,total_edges)

if args.increasing_budget == True:
    draw_results_figures(results_data, decimal, args)
    if args.budget_mode != "attack_number":
        if args.attack_mode in ['graph_guided','graph_guided_cf']:
            draw_figures_contrast(decimal, args)  # compare graph guided feature attack with graph attack
        else:
            draw_figures_contrast_all(decimal, args)  # compare all attack
    else:
        if args.attack_mode != "graph_guided":
            draw_figures_contrast_feature(decimal, args) # compare graph guided and random

'''Record the results'''
if os.path.exists(f'./results_{args.dataset}/result_summary.txt') == False:
    with open(f'./results_{args.dataset}/result_summary.txt', 'w') as f:
        f.write('result record\n')
        line = f"#{args.attack_mode}:lr{args.lr}_epoch{args.attack_epoch}_lamda{args.lamda}_b{args.budget_mode}_c{args.apply_constraint}_s{args.save_opt}_t{args.include_target}_l{args.attack_loss}_{args.random_seed}: \n {results_data['ranking']} \n 5%：{results_data['detect top5%']}"
        f.write(line)
        f.write('\n')
else:
    with open(f'./results_{args.dataset}/result_summary.txt', 'a') as f:
        line = f"#{args.attack_mode}:lr{args.lr}_epoch{args.attack_epoch}_lamda{args.lamda}_b{args.budget_mode}_c{args.apply_constraint}_s{args.save_opt}_t{args.include_target}_l{args.attack_loss}_{args.random_seed}: \n {results_data['ranking']} \n 5%：{results_data['detect top5%']}"
        f.write(line)
        f.write('\n')
