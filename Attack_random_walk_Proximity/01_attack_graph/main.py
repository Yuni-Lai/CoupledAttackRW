# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import random
import pickle
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import argparse
import pprint
from datetime import datetime
from utils import *
from model import *

# Model Settings==============================================================
parser = argparse.ArgumentParser(description='attack RW model')
parser.add_argument('-gpuID', type=int, default=5)
parser.add_argument('-random_seed', type=int, default=2021)
parser.add_argument('-data_size', type=int, default=10000)
parser.add_argument('-similarity_threshold', type=float, default=0.8, help="threshold for constructing proximity graph")
# Attack setting----------------------
parser.add_argument('-lr', type=float, default=1.0, help='learning rate')
parser.add_argument('-lamda', type=float, default=1e-4)
parser.add_argument('-attack_epoch', type=int, default=35, help='training epoch')
parser.add_argument('-target_node_num', type=int, default=20)
parser.add_argument('-random_node_num', type=int, default=100,help='random sample the anomaly nodes from top anomaly nodes')
parser.add_argument('-increasing_budget', action='store_true', default=True,
                    help="increasing attack budget if ture; false:fix the budget")
parser.add_argument('-fixed_budget', type=float, default=0.1,
                    help="see budget_mode for details")
parser.add_argument('-attack_mode', type=str, default='DeepWalk',
                    choices=['alternative', 'closed-form','random','degree','cf-greedy','DeepWalk'])
parser.add_argument('-attack_loss', type=str, default='target_anomaly_sum',
                    choices=['target_anomaly_sum','target_anomaly_adaptive'],
                    help="attacked_graph: make the features be close to the attacked graph, it must used with attack_mode:graph_guided and include_target:False")
parser.add_argument('-budget_mode', type=str, default='node_degree', choices=['totoal_edges', 'node_degree'],
                    help="totoal_edges: total budget = budget * total edges; node_degree : total budget= budget * target nodes * average degree of target node")
parser.add_argument('-scaling', action='store_true', default=False, help="scaling the parameters(B) while optimization")
parser.add_argument('-save_B_opt_loss', action='store_true', default=False, help="save the parameters(B) with lowest loss during training if Ture, else, use the B in final step")
parser.add_argument('-analyze_results', action='store_true', default=False, help="analyze the attack results")
# dir setting----------------------------------------
parser.add_argument('-dataset', type=str, default='Mnist', choices=['KDD99', 'Musk','Satimage','Shuttle','Mnist'])
parser.add_argument('-output_dir', type=str, default='')
args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device(f'cuda:{args.gpuID}')
    print(f"---using GPU---cuda:{args.gpuID}----")
else:
    print("---using CPU---")
    args.device = torch.device("cpu")

if args.attack_mode=="closed-form":
    args.save_B_opt_loss=False
#if args.attack_mode=="random":


if args.dataset == "KDD99":
    args.input_dir = "../DataSets/NetworkIntrusion/KDD99/"
    args.file_name = "kddcup.data_10_percent"
    args.data_size = 10000
    '''Preprocessing '''
    # we stroe the processed data to .pkl so we only need to do it once
    #preprocessing_data(args.input_dir, args.file_name, sample_size=args.data_size)  # sample 1% of anomaly
    args.data_size = args.data_size + int(args.data_size * 0.01)
    ori_features, ori_labels = load_data(args.input_dir)
    ori_labels = np.array([0 if l == "normal" else 1 for l in ori_labels])
    totoal_edge_budget=[round(b, 4) for b in np.arange(0.0001, 0.0006, 0.0001)]
    node_degree_budget=[0.05,0.1,0.2,0.4,0.6,0.8,1.0]
    if args.attack_mode == "closed-form":
        args.lr=0.1

elif args.dataset == "Musk":
    args.input_dir = "../DataSets/Musk"
    args.file_name = "musk.mat"
    args.lr=0.1
    args.lamda=1e-4
    ori_features, ori_labels = load_FromMat(os.path.join(args.input_dir, args.file_name))
    args.data_size = ori_features.shape[0]
    totoal_edge_budget = [round(b, 4) for b in np.arange(0.0001, 0.0013, 0.0002)]#0.002, 0.012, 0.002
    node_degree_budget = [0.05,0.1,0.2,0.4,0.6,0.8,1.0]
    args.similarity_threshold = 0.5
    args.attack_epoch = 80
    args.target_node_num = 5

elif args.dataset == "Mnist":
    args.input_dir = "../DataSets/Mnist"
    args.file_name = "mnist.mat"
    ori_features, ori_labels = load_FromMat(os.path.join(args.input_dir, args.file_name))
    args.data_size = ori_features.shape[0]
    totoal_edge_budget = [round(b, 4) for b in np.arange(0.0001, 0.0013, 0.0002)]#0.002, 0.012, 0.002
    node_degree_budget = [0.05,0.1,0.2,0.4,0.6,0.8,1.0]
    args.similarity_threshold = 0.5
    args.lr = 0.01 #0.01
    args.lamda = 1e-4 #1e-4
    args.attack_epoch = 100
    args.scaling = False
# ============================================================================

ori_features = torch.FloatTensor(ori_features).to(args.device)
n = len(ori_labels)  # number of node
n0 = args.target_node_num  # number of target node


if args.increasing_budget == False:
    budget = [args.fixed_budget]  # budget * average degree = total budget
    args.output_dir = f'./results_{args.dataset}/{args.attack_mode}/e{args.attack_epoch}_lamda{args.lamda}_lr{args.lr}_s{args.scaling}/p{args.fixed_budget}_target{args.target_node_num}_thre{args.similarity_threshold}_{args.random_seed}/'
    results_data = {'anomaly score': [], 'ranking': [], 'detect top1%': [], 'detect top5%': [],
            'detect top10%': [], 'similarity': []}

else:
    if args.budget_mode=="totoal_edges":
        budget = totoal_edge_budget  # budget * total edges = total budget
        decimal = 2  # for drawing figures
    else:
        budget = node_degree_budget  # budget * average degree * target nodes number = total budget ,0.4,0.6,0.8,1.0
        decimal = 1  # for drawing figures
    args.output_dir = f'./results_{args.dataset}/{args.attack_mode}/e{args.attack_epoch}_lamda{args.lamda}_lr{args.lr}_s{args.scaling}/target{args.target_node_num}_thre{args.similarity_threshold}_{args.random_seed}/'
    results_data = {'anomaly score': [], 'ranking': [], 'detect top1%': [], 'detect top5%': [],
            'detect top10%': [], 'similarity': [],
            'budget': []}

print("budget:",budget)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
print("output dir:", args.output_dir)
init_random_seed(args.random_seed)
pprint.pprint(vars(args), width=1)

for similarity_fun in ['correlation','cosine']:  # 'euclidean','correlation','cosine'
    if similarity_fun == 'cosine':
        thre = args.similarity_threshold
    if similarity_fun == 'correlation':
        thre = args.similarity_threshold

    '''Evaluating anomaly scores '''

    similarity_matrix,_,connectivity_scores=get_connectivity_scores(ori_features,similarity_fun,thre)
    # draw_roc(ori_labels, 1-connectivity_scores,title=similarity_fun,savedir=f'{args.output_dir}ROC_{similarity_fun}.pdf')
    #visulize_graph(similarity_matrix)
    '''get the minimum connectivity scores nodes as targets, and sort the data'''
    sorted_index = np.argsort(connectivity_scores,kind='stable') # sort all the data
    features,connectivity_scores,similarity_matrix,labels=sort_features_by_index(sorted_index,ori_features,connectivity_scores,similarity_matrix,ori_labels)
    pseudo_labels = np.array([1]* int(len(ori_features)*0.1) + [0]*(len(ori_features)-int(len(ori_features)*0.1)))
    #random sample the anomaly nodes from top anomaly scores:
    idx=[i for i,l in enumerate(labels) if l==1][0:args.random_node_num]
    random.seed(args.random_seed)
    target_nodes=np.array(random.sample(idx, args.target_node_num))

    target_nodes=target_nodes.astype(int)
    total_edges = (similarity_matrix > 0).sum()/2
    average_degree = np.sum(similarity_matrix>0, axis=0)[target_nodes].mean()# average degree of target nodes
    other_nodes=np.array(list(set(range(args.data_size)) - set(target_nodes)))
    #-----------------------------------------------------

    ranks = rankdata(connectivity_scores)
    print('***Original targets connectivity scores')
    results_data['anomaly score'].append((1 - connectivity_scores)[target_nodes])
    results_data['ranking'].append(ranks[target_nodes].mean() / args.data_size)
    results_data['detect top1%'].append(1-detected(ranks, target_nodes, cutoff=0.01))
    results_data['detect top5%'].append(1-detected(ranks, target_nodes, cutoff=0.05))
    results_data['detect top10%'].append(1-detected(ranks, target_nodes, cutoff=0.1))
    results_data['similarity'].append(similarity_fun)
    if args.increasing_budget == True:
        results_data['budget'].append(0)
    pprint.pprint(results_data)
    print(ranks[target_nodes])

    print(f"start attacking---")
    '''for results analysis'''
    if args.analyze_results:
        analysis_data = {'edge type': [], 'degrees increased': [], 'weights increased': [], 'neighbors increased': [],
                          'attack nodes': [], 'edge number': [], 'budget': [],'nodes degree':[],'anomaly scores':[]}

    if args.scaling == False:
        attacker = Attacker_optimization(features, 0.15, similarity_matrix, connectivity_scores,
                                         target_nodes, similarity_fun, thre, args).to(args.device)
        if args.attack_mode in ['alternative','closed-form']:
            attacker.attack(None)

    if args.attack_mode =='cf-greedy':
        connectivity_scores_atk,results_data=attacker.attack_greedy(budget,average_degree,results_data)
        continue

    for b in budget:
        if args.budget_mode == "totoal_edges":
            total_budget = int(total_edges * b)
        else:
            total_budget = int(average_degree * b * args.target_node_num)
        print(f"budget: {b}, total budget: {total_budget}")
        if args.scaling==True:
            attacker = Attacker_optimization(features, 0.15, similarity_matrix, connectivity_scores,
                                             target_nodes, similarity_fun, thre, args).to(args.device)
            attacker.attack(total_budget)
        if args.attack_mode in ['random', 'degree', 'DeepWalk']:
            connectivity_scores_atk,results_data=attacker.baseline_attack(total_budget,results_data,pseudo_labels)
            if args.increasing_budget == True:
                results_data['budget'].append(b)
            pprint.pprint(results_data)
            print(ranks[target_nodes])
            continue
        connectivity_scores_atk,similarity_matrix_atk,B,topk_B,results_data=attacker.evaluate(b,total_budget,results_data)
        if args.increasing_budget == True:
            results_data['budget'].append(b)
        pprint.pprint(results_data)
        print(ranks[target_nodes])

        if args.analyze_results and similarity_fun=='correlation':
            '''analyze the attack results'''
            analysis_data['budget'].append(b)
            analysis_data['edge number'].append(total_budget)
            analysis_data = analyze_results(analysis_data, B, topk_B, similarity_matrix, similarity_matrix_atk,
                                            target_nodes,connectivity_scores,connectivity_scores_atk)

    if args.analyze_results and similarity_fun=='correlation':
        draw_analysis_figures(analysis_data,decimal,args)

'''visulization'''
if args.increasing_budget==True:
    draw_results_figures(results_data,decimal, args)

'''Record the results'''
if os.path.exists(f'./results_{args.dataset}/result_summary.txt') == False:
    with open(f'./results_{args.dataset}/result_summary.txt', 'w') as f:
        f.write('result record\n')
        if args.increasing_budget == True:
            line = f"{args.attack_mode}_lr{args.lr}_epoch{args.attack_epoch}_lamda{args.lamda}_increasing_b{args.random_seed}: ranking:{results_data['ranking']} \n evasion successful rate_top10%: {results_data['detect top10%']}"
        f.write(line)
        f.write('\n')
else:
    with open(f'./results_{args.dataset}/result_summary.txt', 'a') as f:
        if args.increasing_budget == True:
            line = f"{args.attack_mode}_lr{args.lr}_epoch{args.attack_epoch}_lamda{args.lamda}_increasing_b{args.random_seed}: ranking:{results_data['ranking']} \n evasion successful rate_top10%: {results_data['detect top10%']}"
        f.write(line)
        f.write('\n')