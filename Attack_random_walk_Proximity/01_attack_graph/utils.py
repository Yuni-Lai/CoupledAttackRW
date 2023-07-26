# -*- coding: utf-8 -*-
import random
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import os
import pandas as pd
import pickle
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve,auc
import warnings
import torch
import torch.nn as nn
#import plotly.graph_objects as go
import networkx as nx
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import seaborn as sns;
from matplotlib.patches import Rectangle
sns.set()
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import MultipleLocator
import pprint
plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
plt.style.use('classic')
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['legend.title_fontsize'] = BIGGER_SIZE

def init_random_seed(SEED=2021):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(SEED)
    warnings.filterwarnings("ignore")

def preprocessing_data(path,file_name,sample_size=10000):
    feature_name = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                    'is_hot_login',
                    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                    'srv_rerror_rate',
                    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate',
                    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                    'dst_host_srv_rerror_rate']
    '''we stroe the processed data so we only need to do it once'''
    print("Preprocessing data --------------")
    with open(f"{path}/{file_name}", "r") as f:
        data = f.readlines()

    for i, d in enumerate(data):
        data[i] = d.split(',')

    labels = []
    for i, d in enumerate(data):
        labels.append(d[-1].split('.')[0])
    # counter = collections.Counter(labels)
    # print(counter)
    # {'smurf': 280790, 'neptune': 107201, 'normal': 97278, 'back': 2203, 'satan': 1589, 'ipsweep': 1247,
    #  'portsweep': 1040, 'warezclient': 1020, 'teardrop': 979, 'pod': 264, 'nmap': 231, 'guess_passwd': 53,
    #  'buffer_overflow': 30, 'land': 21, 'warezmaster': 20, 'imap': 12, 'rootkit': 10, 'loadmodule': 9, 'ftp_write': 8,
    #  'multihop': 7, 'phf': 4, 'perl': 3, 'spy': 2}
    normal_data = [data[i] for i, label in enumerate(labels) if label == "normal"]
    anomlay_data = [data[i] for i, label in enumerate(labels) if label != "normal"]
    random.seed(2025)
    data = random.sample(normal_data, sample_size)
    random.seed(2025)
    anomlay_data = random.sample(anomlay_data, int(sample_size * 0.01))
    data.extend(anomlay_data)

    # 19, 20 should be removed!!! all zeros inside. 6 is also 0.
    for i, d in enumerate(data):
        del d[19:21]
        del d[6]

    del feature_name[19:21]
    del feature_name[6]

    dict_name_index = {}
    for i, f in enumerate(feature_name):
        dict_name_index[f] = i

    # Discrete variables
    protocol_type = []
    service = []
    flag = []
    labels = []
    for i, d in enumerate(data):
        protocol_type.append(d[1])
        service.append(d[2])
        flag.append(d[3])
        labels.append(d[-1].split('.')[0])

    discrete_variables = ["protocol_type", "service", "flag"]  # multivalue and nomial/symbolic
    discrete_variables_index = [feature_name.index(v) for v in discrete_variables]
    discrete_variables_elements = []
    for f in discrete_variables:
        discrete_variables_elements.append(set(eval(f)))
        print(f + ':', set(eval(f)))
    # protocol_type: {'tcp', 'icmp', 'udp'}
    # service: {'uucp', 'vmnet', 'uucp_path', 'ctf', 'gopher', 'whois', 'daytime', 'netbios_ns', 'shell', 'sql_net', 'bgp', 'IRC', 'finger', 'systat', 'netbios_ssn', 'tftp_u', 'domain_u', 'pop_3', 'kshell', 'discard', 'http_443', 'nnsp', 'tim_i', 'klogin', 'efs', 'other', 'courier', 'remote_job', 'ftp_data', 'pop_2', 'urp_i', 'red_i', 'ntp_u', 'hostnames', 'csnet_ns', 'smtp', 'imap4', 'eco_i', 'ftp', 'X11', 'rje', 'mtp', 'ssh', 'pm_dump', 'ecr_i', 'login', 'nntp', 'printer', 'iso_tsap', 'sunrpc', 'name', 'Z39_50', 'http', 'telnet', 'auth', 'time', 'private', 'domain', 'netbios_dgm', 'exec', 'urh_i', 'link', 'ldap', 'netstat', 'echo', 'supdup'}
    # flag: {'OTH', 'S0', 'S2', 'S3', 'S1', 'REJ', 'RSTR', 'RSTOS0', 'SF', 'RSTO', 'SH'}

    '''use embedding to represent the discreate data'''
    discrete_variables_index_map = []
    for i, d in enumerate(data):
        for ind1, ind2 in enumerate(discrete_variables_index[::-1]):
            n = len(discrete_variables_elements[::-1][ind1])
            if n <= 5:  # we use one hot vector to represent the discrete variables
                E = np.identity(n).tolist()
            else:  # if the elements number is greater than 5, we use random embedding instead of one-hot
                np.random.seed(2025 + ind1)
                E = np.random.rand(n, 5).tolist()
            dict = {}
            for k, ele in enumerate(discrete_variables_elements[::-1][ind1]):
                dict[ele] = E[k]
            # print(data[i][ind2])
            temp = dict[data[i][ind2]]
            del d[ind2]
            for l in range(min(n, 5)):
                d.insert(ind2 + l, temp[l])  # insert the embedding to the right place.
            if i == 0:
                discrete_variables_index_map.append([ind2, ind2 + min(n - 1, 4)])
                print(
                    f"index of {discrete_variables[ind1]} is {ind2} -> {ind2} ~ {ind2 + min(n - 1, 4)}, len of data + {min(n - 1, 4)}")
        data[i] = [float(value) for value in d[:-1]]

    '''here are the features constraints'''
    integer_variables = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment',
                         'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                         'su_attempted',
                         'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                         'is_guest_login', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
    constraint_range = [[0, 58329], [0, 1379963888], [0, 1309937401], [0, 3], [0, 14], [0, 101], [0, 5], [0, 1],
                        [0, 7479], [0, 1], [0, 1], [0, 7468], [0, 100], [0, 5], [0, 9], [0, 1], [0, 511], [0, 511],
                        [0, 255], [0, 255]]
    dict_name_range = {}
    for i, f in enumerate(integer_variables):
        dict_name_range[f] = constraint_range[i]

    continous_variables = list(set(feature_name) - set(integer_variables) - set(discrete_variables))
    for i, f in enumerate(continous_variables):
        dict_name_range[f] = [0, 1]

    '''index offset'''
    for i, f in enumerate(integer_variables + continous_variables):
        if f == "duration":
            pass
        else:
            dict_name_index[f] = dict_name_index[f] + 10  # 10 is the offset due to embedding of discrete_variables

    len_offset = [0, 2, 6]  # this also aims at dealing with the index offset . the offset is 2+4+4, so that 0,2,6
    for i, f in enumerate(discrete_variables[::-1]):
        # print(f,discrete_variables_index_map[i])
        dict_name_index[f] = [discrete_variables_index_map[i][0] + len_offset[::-1][i],
                              discrete_variables_index_map[i][1] + len_offset[::-1][i]]
    pprint.pprint(dict_name_index)

    '''save to files'''
    f = open(f'{path}/data.pkl', 'wb')
    pickle.dump([data, labels], f)
    f.close()
    print("---------------------------------")

    print("###make these discrete variables unchanged:###")
    for v in discrete_variables:
        print(v, dict_name_index[v])
    print("###make these discrete variables in a certain range and integer:###")
    for v in integer_variables:
        print(f'{v},{dict_name_index[v]},{dict_name_range[v]}')
    print("###make the other continous variables in a certain range also###")
    for v in continous_variables:
        print(f'{v},{dict_name_index[v]},{dict_name_range[v]}')

    variable_types = {"discrete_variables": discrete_variables, "integer_variables": integer_variables,
                      "continous_variables": continous_variables}

    f = open(f'{path}/data_constraint_info.pkl', 'wb')
    pickle.dump([dict_name_index, dict_name_range, variable_types], f)
    f.close()

def load_data(path):
    f=open(f'{path}/data.pkl','rb')
    features,labels=pickle.load(f)
    f.close()
    return features,labels

def load_FromMat(path):
    import scipy.io as scio
    dict1 = scio.loadmat(path)
    feature=dict1['X']
    label=dict1['y']
    return feature,label

def construct_line_graph(adj_matrix):
    """Construct a line graph from an undirected original graph.
    Parameters
    ----------
    adj_matrix : sp.spmatrix [n_samples ,n_samples]
        Symmetric binary adjacency matrix.
    Returns
    -------
    L : sp.spmatrix, shape [A.nnz/2, A.nnz/2]
        Symmetric binary adjacency matrix of the line graph.
    """
    N = adj_matrix.shape[0]
    edges = np.column_stack(sp.triu(adj_matrix, 1).nonzero())
    e1, e2 = edges[:, 0], edges[:, 1]
    I = sp.eye(N,dtype='float16').tocsr()
    #I = np.eye(N, dtype='float16')
    E1 = I[e1]
    E2 = I[e2]
    L = E1.dot(E1.T)+ E2.dot(E1.T) + E1.dot(E2.T) + E2.dot(E2.T)
    return L - 2 * sp.eye(L.shape[0])

def generate_candidates_removal(adj_matrix, seed=0):
    """Generates candidate edge flips for removal (edge -> non-edge),
     disallowing one random edge per node to prevent singleton nodes.

    adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    """
    n_nodes = adj_matrix.shape[0]

    np.random.seed(seed)
    deg = np.where(adj_matrix.sum(1).A1 == 1)[0]

    hiddeen = np.column_stack(
        (np.arange(n_nodes), np.fromiter(map(np.random.choice, adj_matrix.tolil().rows), dtype=np.int)))

    adj_hidden = edges_to_sparse(hiddeen, adj_matrix.shape[0])
    adj_hidden = adj_hidden.maximum(adj_hidden.T)

    adj_keep = adj_matrix - adj_hidden

    candidates = np.column_stack((sp.triu(adj_keep).nonzero()))

    candidates = candidates[np.logical_not(np.in1d(candidates[:, 0], deg) | np.in1d(candidates[:, 1], deg))]

    return candidates


def generate_candidates_addition(adj_matrix, targets):
    """Generates candidate edge flips for addition (non-edge -> edge).

    adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param n_candidates: int
        Number of candidates to generate.
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    """
    num_nodes = adj_matrix.shape[0]
    nontargets = [n for n in range(0,num_nodes) if n not in targets]
    candidates_list=[]
    for item in itertools.product(targets, nontargets):
        if adj_matrix[item[0],item[1]] ==0:
            candidates_list.append(item)
    candidates = np.array(candidates_list)
    return candidates


def draw_roc(labels, connectivity_scores,title=None,savedir='./'):
    fpr, tpr, thresholds = roc_curve(labels, connectivity_scores, pos_label=None, sample_weight=None,
                                     drop_intermediate=None)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(constrained_layout=True)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC = %0.2f' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title(f'Receiver operating curve ({title})', fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(savedir, dpi=300)
    plt.show()

def sort_features_by_index(sorted_index,features,connectivity_scores,similarity_matrix,labels):
    #features = torch.FloatTensor(features).to(args.device)
    features = features[sorted_index]
    connectivity_scores = connectivity_scores[sorted_index]
    similarity_matrix = similarity_matrix[:, sorted_index]
    similarity_matrix = similarity_matrix[sorted_index, :]
    labels = labels[sorted_index]
    return features,connectivity_scores,similarity_matrix,labels

def largest_indices(tensor, n):
    """Returns the n largest indices from a numpy array."""
    ary = tensor.cpu().numpy()
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices],kind='stable')]
    return np.unravel_index(indices, ary.shape)

def detected(ranks,target_nodes,cutoff=0.01):
    total=len(ranks)
    anomaly_num=int(total*cutoff)
    detected_nodes=[n for n in target_nodes if ranks[n]<anomaly_num]
    detected_percentage=len(detected_nodes)/target_nodes.shape[0]
    return detected_percentage

def visulize_graph(A):
    G=nx.from_numpy_matrix(A)# graph is too big to load

def visulize_graph(A):
    G=nx.from_numpy_matrix(A)# graph is too big to load

def analyze_results(analysis_data,B,topk_B,matrix_ori,matrix_atk,target_nodes,connectivity_scores_ori,connectivity_scores_atk):
    atked_edges = torch.tensor(topk_B.cpu().numpy().nonzero())
    atked_nodes = torch.unique(atked_edges) #atked_nodes = np.setdiff1d(np.array(atked_nodes), target_nodes)
    other_nodes = np.setdiff1d(range(0, B.shape[0]),np.array(atked_nodes))
    print(f"attack nodes: {atked_nodes.shape[0]}")

    '''1) the changes of target nodes degree'''
    unweighted_degree=np.sum(matrix_ori>0,axis=1)
    weighted_degree=np.sum(matrix_ori,axis=1)
    degree_ori=unweighted_degree[target_nodes]
    degree_atk=np.sum(matrix_atk>0,axis=1)[target_nodes]
    degree_increased=degree_atk-degree_ori
    print(f"The attack increased the degree of target nodes:{degree_increased}")

    '''2) the changes of weights'''
    Boolean = (topk_B.cpu().numpy().nonzero())
    edge_weight_ori = matrix_ori[Boolean]
    edge_weight_atk = matrix_atk[Boolean]
    weight_increased=edge_weight_atk - edge_weight_ori
    print(f"edge weights increased after attack:{weight_increased}")

    '''3) the changes of 1-hops neighbors'''
    edge_index_ori = torch.tensor(matrix_ori.nonzero())

    if "original 1-hop neighbors" not in analysis_data:
        subset1, _, _, _ = k_hop_subgraph(
            list(target_nodes), 1, edge_index_ori, relabel_nodes=False)
        print("the number of original 1-hops neighbor:", subset1.shape[0])
        # subset2, _, _, _ = k_hop_subgraph(
        #     list(target_nodes), 2, edge_index_ori, relabel_nodes=False)
        # print("the number of original 2-hops neighbor:", subset2.shape[0])
        analysis_data["original 1-hop neighbors"]=subset1.shape[0]

    edge_index_atk = torch.tensor(matrix_atk.nonzero())
    subset1_atk, _, _, _ = k_hop_subgraph(
        list(target_nodes), 1, edge_index_atk, relabel_nodes=False)
    neighbors_increased=subset1_atk.shape[0] - analysis_data["original 1-hop neighbors"]
    print("the number of attacked 1-hops neighbor:", subset1_atk.shape[0])
    print(f"The attack increased the one-hop neighbors of target nodes:{neighbors_increased}")
    
    '''4) where the atked_edges are added'''
    edge_types={'target-target':0,'target-other':0,'others-other':0}
    atked_edges=atked_edges.numpy()
    for index in range(atked_edges.shape[1]):
        i,j=atked_edges[:,index]
        if i in target_nodes and j in target_nodes:
            edge_types['target-target']=edge_types['target-target']+1
        elif i not in target_nodes and j not in target_nodes:
            edge_types['others-other'] = edge_types['others-other'] + 1
        else:
            edge_types['target-other'] = edge_types['target-other'] + 1

    analysis_data['attack nodes'].append(atked_nodes.shape[0]/B.shape[0])
    analysis_data['degrees increased'].append(degree_increased)
    analysis_data['weights increased'].append(weight_increased)
    analysis_data['neighbors increased'].append(neighbors_increased)
    analysis_data['edge type'].append(edge_types)
    analysis_data['nodes degree'].append({'atked_nodes':weighted_degree[atked_nodes],'other_nodes':weighted_degree[other_nodes]})
    anomaly_dict={'target_nodes':1-connectivity_scores_ori[target_nodes],'atked_nodes':1-connectivity_scores_ori[atked_nodes],'other_nodes':1-connectivity_scores_ori[other_nodes]}
    analysis_data['anomaly scores'].append(anomaly_dict)
    return analysis_data

def draw_results_figures(data,decimal,args):
    df = pd.DataFrame(data)
    df.head()
    print(df)
    df = df.rename(columns={"similarity function": "similarity"})

    anomaly_scores = []
    for j in range(len(data['anomaly score'][0])):
        for i in range(len(data['budget'])):
            anomaly_scores.append(data['anomaly score'][i][j])
    df_anomaly = pd.DataFrame(
        {'budget': data['budget'] * len(data['anomaly score'][0]),'similarity':data['similarity'] * len(data['anomaly score'][0]), 'anomaly score': anomaly_scores})

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='anomaly score', hue='similarity', data=df_anomaly, legend='full', linewidth=2.0)
    plt.ylabel('anomaly score')
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4e'))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=decimal))
    if decimal>=3:
        plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'GraphAttackResult_anomaly_score.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='ranking', hue='similarity', data=df, legend='full', linewidth=2.0)
    plt.ylabel('anomaly ranking')
    plt.legend(loc='lower right')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=decimal))
    if decimal>=3:
        plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'GraphAttackResult_ranking.pdf', dpi=300)
    plt.show()

    df_melted = pd.melt(df,
                        id_vars=['budget','similarity'],
                        value_vars=['budget','similarity', 'detect top1%', 'detect top5%','detect top10%'],
                        var_name=["threshold"],
                        value_name='data')

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='data', hue='threshold',style='similarity', data=df_melted, legend='full', linewidth=2.0)
    plt.ylabel('evasion successful rate')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    if decimal >= 3:
        plt.xticks(rotation=45)
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.legend(loc='lower right')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    plt.savefig(args.output_dir + 'GraphAttackResult_EvasionRate.pdf', dpi=300)
    plt.show()

    print("results(figures) saved to:", args.output_dir)
    f = open(args.output_dir + '/attack_graph_results.pkl', 'wb')
    pickle.dump([df, df_anomaly,df_melted], f)
    f.close()

def draw_analysis_figures(analysis_data,decimal,args):
    df = pd.DataFrame({'budget': analysis_data['budget'], 'edge number': analysis_data['edge number'],'neighbors increased': analysis_data['neighbors increased'],'attack nodes': analysis_data['attack nodes']})

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='edge number', data=df, legend='full', linewidth=2.0, marker='o')
    plt.ylabel('edges number')
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    if args.budget_mode == "totoal_edges":
        x_major_locator = MultipleLocator(0.0001)
    else:
        x_major_locator = MultipleLocator(0.2)
    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    #plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'AnalysisEdgeNumber.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='neighbors increased', data=df, legend='full', linewidth=2.0, marker='o')
    plt.ylabel('1-hop neighbors increased')
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    #plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'AnalysisNeighbor.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='attack nodes', data=df, legend='full', linewidth=2.0, marker='o')
    plt.ylabel('attack nodes proportion')
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    #plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'AnalysisAttackNumber.pdf', dpi=300)
    plt.show()

    print("results(figures) saved to:", args.output_dir)
    f = open(args.output_dir + '/analysis_results.pkl', 'wb')
    pickle.dump(df, f)
    f.close()


    df = pd.DataFrame({'budget': analysis_data['budget'], 'target-target': [d['target-target'] for d in analysis_data['edge type']],'target-other':[d['target-other'] for d in analysis_data['edge type']],'others-other':[d['others-other'] for d in analysis_data['edge type']]})
    df_melted = pd.melt(df,
                        id_vars='budget',
                        value_vars=['budget', 'target-target', 'target-other','others-other'],
                        var_name='edge types',
                        value_name='data')

    plt.figure(constrained_layout=True)
    df_melted['budget']=[ str(round(x*100,decimal))+"%" for x in df_melted['budget']]
    sns.barplot(x='budget', y='data', hue="edge types", data=df_melted)
    plt.ylabel('edge number')
    if decimal >= 3:
        plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'AnalysisEdge.pdf', dpi=300)
    plt.show()


    data=[]
    budget=[]
    for i in range(len(analysis_data['budget'])):
        for j in range(len(analysis_data['degrees increased'][i])):
            data.append(analysis_data['degrees increased'][i][j])
            budget.append(analysis_data['budget'][i])
    df = pd.DataFrame(
        {'budget': budget, 'degrees increased':data})

    plt.figure(constrained_layout=True)
    df['budget'] = [str(round(x * 100, decimal)) + "%" for x in df['budget']]
    sns.barplot(x='budget', y='degrees increased', data=df,palette="Blues",linewidth=1)
    plt.ylabel('target degrees increased')
    if decimal >= 3:
        plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'AnalysisDegree.pdf', dpi=300)
    plt.show()

    data=[]
    budget = []
    for i in range(len(analysis_data['budget'])):
        for j in range(len(analysis_data['weights increased'][i])):
            data.append(analysis_data['weights increased'][i][j])
            budget.append(analysis_data['budget'][i])
    df = pd.DataFrame(
        {'budget': budget, 'weights increased':data})
    df['budget'] = [str(round(x * 100, decimal)) + "%" for x in df['budget']]
    plt.figure(constrained_layout=True)
    sns.stripplot(x='budget', y='weights increased', data=df,palette="Blues",jitter=1)
    plt.ylabel('weights increased')
    if decimal >= 3:
        plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'AnalysisWeight.pdf', dpi=300)
    plt.show()


    plt.figure(constrained_layout=True)
    plt.title(f"Original anomaly scores (budget: {round(analysis_data['budget'][2]*100, decimal)}%)")
    plt.hist(analysis_data['anomaly scores'][2]['other_nodes'], bins=50, facecolor="darkorange", edgecolor="black", alpha=0.5)
    plt.hist(analysis_data['anomaly scores'][2]['atked_nodes'], bins=25, facecolor="sienna", edgecolor="black", alpha=0.7)
    plt.hist(analysis_data['anomaly scores'][2]['target_nodes'], bins=1, facecolor="red", edgecolor="black", alpha=1.0)
    target = "red"
    attack = "sienna"
    other = "darkorange"
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k", alpha=0.5) for c in [attack, other, target]]
    labels_list = ["attack nodes", "other nodes", "target nodes"]
    plt.legend(handles, labels_list, loc='upper left')
    plt.ylabel("count")
    plt.xlabel("anomaly scores")
    plt.gca().ticklabel_format(style='sci', axis='x',scilimits=(0,0),useMathText=True)
    #plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.4e'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_dir + 'AnalysisAnomaly.pdf', dpi=300)
    plt.show()

    budget_list=[]
    types_list=[]
    data=[]
    mapping={'atked_nodes':"attack nodes",'other_nodes':"other nodes"}
    for i in range(len(analysis_data['budget'])):
        for type in ['atked_nodes', 'other_nodes']:
            for d in analysis_data['nodes degree'][i][type]:
                budget_list.append(analysis_data['budget'][i])
                data.append(d)
                types_list.append(mapping[type])
    df=pd.DataFrame({'budget':budget_list,'nodes degree':data,'types':types_list})
    df['budget'] = [str(round(x * 100, decimal)) + "%" for x in df['budget']]
    plt.figure(constrained_layout=True)
    sns.boxplot(x='budget', y='nodes degree',hue='types', data=df)
    plt.gca().legend_.set_title(None)
    plt.legend(loc='lower right')
    plt.savefig(args.output_dir + 'AnalysisAttackDegree.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    plt.title(f"Original nodes degree (budget: {round(analysis_data['budget'][2] * 100, decimal)}%)")
    plt.hist(analysis_data['nodes degree'][2]['other_nodes'], bins=50, facecolor="darkorange", edgecolor="black",
             alpha=0.5)
    plt.hist(analysis_data['nodes degree'][2]['atked_nodes'], bins=5, facecolor="sienna", edgecolor="black",
             alpha=0.7)
    attack = "sienna"
    other = "darkorange"
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k", alpha=0.5) for c in [attack, other]]
    labels_list = ["attack nodes", "other nodes"]
    plt.legend(handles, labels_list, loc='upper left')
    plt.ylabel("count")
    plt.xlabel("nodes degree before attack")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_dir + 'AnalysisNodeDegree.pdf', dpi=300)
    plt.show()

if __name__ == '__main__':
    data_path = "DataSets/NetworkIntrusion/KDD99/kddcup.data_10_percent"
    preprocessing_data(data_path,sample_size=10000)