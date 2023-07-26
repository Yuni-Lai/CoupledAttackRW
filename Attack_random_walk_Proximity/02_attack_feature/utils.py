# -*- coding: utf-8 -*-
import random
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import os
import pandas
import collections
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve,auc,precision_score
import warnings
import torch
import pprint
import torch.nn as nn
import pandas as pd
import seaborn as sns;
sns.set()
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import MultipleLocator
import argparse

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
    feature_name=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
                  'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
                  'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login',
                  'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
                  'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
                  'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
                  'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
    '''we stroe the processed data so we only need to do it once'''
    print("Preprocessing data --------------")
    with open(f"{path}/{file_name}", "r") as f:
        data = f.readlines()

    for i, d in enumerate(data):
        data[i] = d.split(',')

    labels = []
    for i, d in enumerate(data):
        labels.append(d[-1].split('.')[0])
    #counter = collections.Counter(labels)
    #print(counter)
    # {'smurf': 280790, 'neptune': 107201, 'normal': 97278, 'back': 2203, 'satan': 1589, 'ipsweep': 1247,
    #  'portsweep': 1040, 'warezclient': 1020, 'teardrop': 979, 'pod': 264, 'nmap': 231, 'guess_passwd': 53,
    #  'buffer_overflow': 30, 'land': 21, 'warezmaster': 20, 'imap': 12, 'rootkit': 10, 'loadmodule': 9, 'ftp_write': 8,
    #  'multihop': 7, 'phf': 4, 'perl': 3, 'spy': 2}
    normal_data = [data[i] for i, label in enumerate(labels) if label == "normal"]
    anomlay_data = [data[i] for i, label in enumerate(labels) if label != "normal"]
    random.seed(2025)
    data = random.sample(normal_data, sample_size)
    random.seed(2025)
    anomlay_data = random.sample(anomlay_data, int(sample_size*0.01))
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

    discrete_variables = ["protocol_type", "service", "flag"]# multivalue and nomial/symbolic
    discrete_variables_index = [feature_name.index(v) for v in discrete_variables]
    discrete_variables_elements = []
    for f in discrete_variables:
        discrete_variables_elements.append(set(eval(f)))
        print(f + ':', set(eval(f)))
    # protocol_type: {'tcp', 'icmp', 'udp'}
    # service: {'uucp', 'vmnet', 'uucp_path', 'ctf', 'gopher', 'whois', 'daytime', 'netbios_ns', 'shell', 'sql_net', 'bgp', 'IRC', 'finger', 'systat', 'netbios_ssn', 'tftp_u', 'domain_u', 'pop_3', 'kshell', 'discard', 'http_443', 'nnsp', 'tim_i', 'klogin', 'efs', 'other', 'courier', 'remote_job', 'ftp_data', 'pop_2', 'urp_i', 'red_i', 'ntp_u', 'hostnames', 'csnet_ns', 'smtp', 'imap4', 'eco_i', 'ftp', 'X11', 'rje', 'mtp', 'ssh', 'pm_dump', 'ecr_i', 'login', 'nntp', 'printer', 'iso_tsap', 'sunrpc', 'name', 'Z39_50', 'http', 'telnet', 'auth', 'time', 'private', 'domain', 'netbios_dgm', 'exec', 'urh_i', 'link', 'ldap', 'netstat', 'echo', 'supdup'}
    # flag: {'OTH', 'S0', 'S2', 'S3', 'S1', 'REJ', 'RSTR', 'RSTOS0', 'SF', 'RSTO', 'SH'}

    '''use embedding to represent the discreate (nominal) data'''
    discrete_variables_index_map=[]
    for i, d in enumerate(data):
        for ind1, ind2 in enumerate(discrete_variables_index[::-1]):
            n = len(discrete_variables_elements[::-1][ind1])
            if n <= 5:  # we use one hot vector to represent the discrete variables
                E = np.identity(n).tolist()
            else:  # if the elements number is greater than 5, we use random embedding instead of one-hot
                np.random.seed(2025+ind1)
                E = np.random.rand(n, 5).tolist()
            dict = {}
            for k, ele in enumerate(discrete_variables_elements[::-1][ind1]):
                dict[ele] = E[k]
            # print(data[i][ind2])
            temp = dict[data[i][ind2]]
            del d[ind2]
            for l in range(min(n, 5)):
                d.insert(ind2 + l, temp[l])# insert the embedding to the right place.
            if i==0:
                discrete_variables_index_map.append([ind2,ind2+min(n-1, 4)])
                print(f"index of {discrete_variables[ind1]} is {ind2} -> {ind2} ~ {ind2+min(n-1, 4)}, len of data + {min(n-1, 4)}")
        data[i] = [float(value) for value in d[:-1]]

    '''here are the features constraints'''
    integer_variables = ['duration','src_bytes','dst_bytes','wrong_fragment',
                  'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
                  'num_root','num_file_creations','num_shells','num_access_files',
                  'is_guest_login','count','srv_count','dst_host_count','dst_host_srv_count']
    constraint_range=[[0,58329],[0,1379963888],[0,1309937401],[0,3],[0,14],[0,101],[0,5],[0,1],[0,7479],[0,1],[0,1],[0,7468],[0,100],[0,5],[0,9],[0,1],[0,511],[0,511],[0,255],[0,255]]
    dict_name_range = {}
    for i, f in enumerate(integer_variables):
        dict_name_range[f] = constraint_range[i]

    continous_variables = list(set(feature_name)-set(integer_variables)-set(discrete_variables))
    for i, f in enumerate(continous_variables):
        dict_name_range[f] = [0,1]

    '''index offset'''
    for i,f in enumerate(integer_variables+continous_variables):
        if f=="duration":
            pass
        else:
            dict_name_index[f]=dict_name_index[f]+10# 10 is the offset due to embedding of discrete_variables

    len_offset=[0,2,6] # this also aims at dealing with the index offset . the offset is 2+4+4, so that 0,2,6
    for i,f in enumerate(discrete_variables[::-1]):
        #print(f,discrete_variables_index_map[i])
        dict_name_index[f] = [discrete_variables_index_map[i][0]+len_offset[::-1][i],discrete_variables_index_map[i][1]+len_offset[::-1][i]]
    pprint.pprint(dict_name_index)

    '''save to files'''
    f = open(f'{path}/data.pkl', 'wb')
    pickle.dump([data, labels], f)
    f.close()
    print("---------------------------------")

    print("###make these discrete variables unchanged:###")
    for v in discrete_variables:
        print(v,dict_name_index[v])
    print("###make these discrete variables in a certain range and integer:###")
    for v in integer_variables:
        print(f'{v},{dict_name_index[v]},{dict_name_range[v]}')
    print("###make the other continous variables in a certain range also###")
    for v in continous_variables:
        print(f'{v},{dict_name_index[v]},{dict_name_range[v]}')

    variable_types={"discrete_variables":discrete_variables,"integer_variables":integer_variables,"continous_variables":continous_variables}

    f = open(f'{path}/data_constraint_info.pkl', 'wb')
    pickle.dump([dict_name_index,dict_name_range,variable_types], f)
    f.close()

def load_data(path):
    f=open(f'{path}/data.pkl','rb')
    features,labels=pickle.load(f)
    f.close()
    return features,labels

def detected(ranks,target_nodes,cutoff=0.01):
    total=len(ranks)
    anomaly_num=int(total*cutoff)
    detected_nodes=[n for n in target_nodes if ranks[n]<anomaly_num]
    detected_percentage=len(detected_nodes)/target_nodes.shape[0]
    return detected_percentage

def choose_control_nodes(args,target_nodes,nontarget_nodes,similarity_fun,b,similarity_matrix_1):
    if args.budget_mode != "attack_number":
        with open(args.graph_dir + f'/B_{similarity_fun}_{b}.pth', 'rb') as f:
            B, topk_B, target_nodes_graph = pickle.load(f)
            assert (np.array(target_nodes_graph.cpu()) == target_nodes).all()
        edges = torch.tensor(topk_B.cpu().numpy().nonzero())
        atked_nodes = torch.unique(edges)
        atked_nodes = torch.unique(torch.cat([atked_nodes, torch.LongTensor(target_nodes)]))
        total_budget = len(atked_nodes)
    else:  # budget mode: attack_number
        with open(args.graph_dir + f'/B_{similarity_fun}_1.0.pth', 'rb') as f:
            B, topk_B, target_nodes_graph = pickle.load(f)
            assert (np.array(target_nodes_graph.cpu()) == target_nodes).all()
        edges = torch.tensor(topk_B.cpu().numpy().nonzero())
        atked_nodes = torch.unique(edges)
        total_budget = b

    if args.attack_mode == "random":
        random.seed(args.random_seed)
        if args.include_target == True:
            controlled_nodes = np.append(target_nodes, np.array(
                random.sample(nontarget_nodes, total_budget - args.target_node_num)))
        else:
            controlled_nodes = np.array(random.sample(nontarget_nodes, total_budget))
    elif args.attack_mode == "random_low_degree":
        unweighted_degree = np.sum(similarity_matrix_1 > 0, axis=1)
        degree_threshold = np.percentile(unweighted_degree, 30)
        random.seed(args.random_seed)
        if args.include_target == True:
            controlled_nodes = np.append(target_nodes, np.array(
                random.sample([n for n in nontarget_nodes if unweighted_degree[n] <= degree_threshold],
                              total_budget - args.target_node_num)))
        else:
            controlled_nodes = np.array(
                random.sample([n for n in nontarget_nodes if unweighted_degree[n] <= degree_threshold], total_budget))
    elif args.attack_mode == "random_high_degree":
        unweighted_degree = np.sum(similarity_matrix_1 > 0, axis=1)
        degree_threshold = np.percentile(unweighted_degree, 70)
        random.seed(args.random_seed)
        if args.include_target == True:
            controlled_nodes = np.append(target_nodes, np.array(
                random.sample([n for n in nontarget_nodes if unweighted_degree[n] >= degree_threshold],
                              total_budget - args.target_node_num)))
        else:
            controlled_nodes = np.array(
                random.sample([n for n in nontarget_nodes if unweighted_degree[n] >= degree_threshold], total_budget))
    elif args.budget_mode != "attack_number": # attack_mode: graph-guided;
        if args.include_target == True:
            controlled_nodes = np.unique(np.append(target_nodes, np.array(atked_nodes)))
        else:
            controlled_nodes = np.setdiff1d(np.array(atked_nodes), target_nodes)
    else:  # attack_mode: graph-guided; budget mode: attack_number
        random.seed(args.random_seed)
        if args.include_target == True:
            controlled_nodes = np.append(target_nodes,
                                         random.sample(list(np.setdiff1d(atked_nodes.numpy(), target_nodes)),
                                                       total_budget - args.target_node_num))
        else:
            controlled_nodes = np.array(
                random.sample(list(np.setdiff1d(atked_nodes.numpy(), target_nodes)), total_budget))
    return controlled_nodes,topk_B


def load_FromMat(path):
    import scipy.io as scio
    dict1 = scio.loadmat(path)
    feature=dict1['X']
    label=dict1['y']
    return feature,label

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
    plt.xlabel('false Positive Rate', fontsize=20)
    plt.ylabel('true Positive Rate', fontsize=20)
    plt.title(f'receiver operating curve ({title})', fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(savedir, dpi=300)
    plt.show()

def analyze_results(b,analysis_data,controlled_nodes,similarity_matrix, similarity_matrix_atk,target_nodes,total_edges):
    unweighted_degree = np.sum(similarity_matrix > 0, axis=1)
    analysis_data['budget'].append(b)
    edge_modified = np.sum(np.round(similarity_matrix_atk,3) != np.round(np.float32(similarity_matrix),3))/(2*total_edges)
    analysis_data['edge_modified'].append(edge_modified)
    control_degrees=unweighted_degree[controlled_nodes]
    analysis_data['control_degrees'].append(control_degrees)
    #[d for d in analysis_data['control_degrees'][0] if d not in control_degrees]
    # edge_index = torch.tensor(similarity_matrix.nonzero())
    # subset2, _, _, _ = k_hop_subgraph(
    #     list(target_nodes), 2, edge_index, relabel_nodes=False)
    # print("the number of 2-hops neighbor:", subset2.shape)
    # subset1, _, _, _ = k_hop_subgraph(
    #     list(target_nodes), 1, edge_index, relabel_nodes=False)
    # print("the number of 1-hops neighbor:", subset1.shape)
    return analysis_data

def draw_results_figures(data, decimal, args):
    df = pd.DataFrame(data)
    df.head()
    print(df)

    anomaly_scores = []
    for j in range(len(data['anomaly score'][0])):
        for i in range(len(data['budget'])):
            anomaly_scores.append(data['anomaly score'][i][j])
    df_anomaly = pd.DataFrame(
        {'budget': data['budget'] * len(data['anomaly score'][0]),
         'similarity': data['similarity'] * len(data['anomaly score'][0]),
         'anomaly score': anomaly_scores})

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='anomaly score', hue='similarity', data=df_anomaly, legend='full',
                 linewidth=2.0)
    plt.ylabel('anomaly score')
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4e'))
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'FeatureAttackResult_anomaly_score.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='ranking', hue='similarity', data=df, legend='full', linewidth=2.0)
    plt.ylabel('anomaly ranking')
    plt.legend(loc='lower right')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(args.output_dir + 'FeatureAttackResult_ranking.pdf', dpi=300)
    plt.show()

    df_melted = pd.melt(df,
                        id_vars=['budget', 'similarity'],
                        value_vars=['budget', 'similarity', 'detect top1%', 'detect top5%',
                                    'detect top10%'],
                        var_name=["threshold"],
                        value_name='data')
    # df_melted = df_melted.rename(columns={"similarity": "similarity"})
    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='data', hue='threshold', style='similarity', data=df_melted, legend='full',
                 linewidth=2.0)
    plt.ylabel('evasion successful rate')
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.legend(loc='lower right')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    plt.savefig(args.output_dir + 'FeatureAttackResult_EvasionRate.pdf', dpi=300)
    plt.show()

    print("results(figures) saved to:", args.output_dir)
    f = open(args.output_dir + '/attack_feature_results.pkl', 'wb')
    pickle.dump([df, df_anomaly, df_melted], f)
    f.close()

def draw_figrues(df_anomaly_temp,df_temp,df_melted_temp,args,decimal):

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='anomaly score', hue='attack types', style='similarity', data=df_anomaly_temp,
                 legend='full', linewidth=2.0)
    plt.ylabel('anomaly score')
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(args.output_dir + f'Contrast_anomaly_score_constraint{args.apply_constraint}.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='ranking', hue='attack types', style='similarity', data=df_temp,
                 legend='full', linewidth=2.0)
    plt.ylabel('anomaly ranking')
    plt.legend(loc='upper left')
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(args.output_dir + f'Contrast_ranking_constraint{args.apply_constraint}.pdf', dpi=300)
    plt.show()

    df_cor = df_melted_temp[df_melted_temp["similarity"] == "correlation"]
    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='data', hue="attack types", style='threshold', markers=True, data=df_cor,
                 legend='full',
                 linewidth=2.0)
    plt.ylabel('evasion successful rate')
    plt.legend(loc='lower left')
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(args.output_dir + f'Contrast_evasion_constraint{args.apply_constraint}_correlation.pdf', dpi=300)
    plt.show()

    df_cor=df_cor[df_cor["threshold"] == "detect top10%"]
    plt.figure(constrained_layout=True)
    sns.barplot(x='budget', y='data', hue="attack types",  data=df_cor,palette="Paired")
    plt.ylabel('evasion successful rate (top 10%)')
    plt.legend(loc='upper right')
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(args.output_dir + f'Contrast_evaTop5_constraint{args.apply_constraint}_correlation.pdf', dpi=300)
    plt.show()

    df_cos = df_melted_temp[df_melted_temp["similarity"] == "cosine"]
    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='data', hue="attack types", style='threshold', markers=True, data=df_cos,
                 legend='full',
                 linewidth=2.0)
    plt.ylabel('evasion successful rate')
    plt.legend(loc='lower left')
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(args.output_dir + f'Contrast_evasion_constraint{args.apply_constraint}_cosine.pdf', dpi=300)
    plt.show()

    df_cos=df_cos[df_cos["threshold"] == "detect top5%"]
    plt.figure(constrained_layout=True)
    sns.barplot(x='budget', y='data', hue="attack types",  data=df_cos,palette="Paired")
    plt.ylabel('evasion successful rate (top 5%)')
    plt.legend(loc='upper right')
    if args.budget_mode != "attack_number":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
        if decimal >= 3:
            plt.xticks(rotation=45)
    plt.savefig(args.output_dir + f'Contrast_evaTop5_constraint{args.apply_constraint}_cosine.pdf', dpi=300)
    plt.show()

def draw_figures_contrast(decimal,args):
    f = open(args.output_dir + '/attack_feature_results.pkl', 'rb')
    df, df_anomaly, df_melted=pickle.load(f)
    f.close()

    f = open(args.graph_dir + '/attack_graph_results.pkl', 'rb')
    df_2, df_anomaly_2, df_melted_2 = pickle.load(f)
    f.close()

    df_anomaly["attack types"]="feature attack"
    df_anomaly_2["attack types"] = "graph attack"
    df_anomaly_temp=pd.concat([df_anomaly,df_anomaly_2], ignore_index=True)

    df["attack types"] = "feature attack"
    df_2["attack types"] = "graph attack"
    df_temp = pd.concat([df, df_2], ignore_index=True)

    df_melted["attack types"] = "feature attack"
    df_melted_2["attack types"] = "graph attack"
    df_melted_temp = pd.concat([df_melted, df_melted_2], ignore_index=True)

    draw_figrues(df_anomaly_temp, df_temp, df_melted_temp, args,decimal)

def draw_figures_contrast_all(decimal, args):
    f = open(args.output_dir + '/attack_feature_results.pkl', 'rb')
    df, df_anomaly, df_melted = pickle.load(f)
    f.close()

    dir=f'./results_{args.dataset}/e{args.attack_epoch}_ld{args.lamda}_lr{args.lr}_mgraph_guided_c{args.apply_constraint}/b{args.budget_mode}_al{args.attack_loss}_s{args.save_opt}_{args.random_seed}/'
    f = open(dir + '/attack_feature_results.pkl', 'rb')
    df_2, df_anomaly_2, df_melted_2 = pickle.load(f)
    f.close()

    f = open(args.graph_dir + '/attack_graph_results.pkl', 'rb')
    df_3, df_anomaly_3, df_melted_3 = pickle.load(f)
    f.close()

    df_anomaly["attack types"] = "random"
    df_anomaly_2["attack types"] = "graph guided"
    df_anomaly_3["attack types"] = "graph attack"
    df["attack types"] = "random"
    df_2["attack types"] = "graph guided"
    df_3["attack types"] = "graph attack"
    df_melted["attack types"] = "random"
    df_melted_2["attack types"] = "graph guided"
    df_melted_3["attack types"] = "graph attack"

    df_anomaly_temp = pd.concat([df_anomaly, df_anomaly_2,df_anomaly_3], ignore_index=True)
    df_temp = pd.concat([df, df_2,df_3], ignore_index=True)
    df_melted_temp = pd.concat([df_melted, df_melted_2, df_melted_3], ignore_index=True)
    draw_figrues(df_anomaly_temp, df_temp, df_melted_temp, args,decimal)

def draw_figures_contrast_feature(decimal, args):
    f = open(args.output_dir + '/attack_feature_results.pkl', 'rb')
    df, df_anomaly, df_melted = pickle.load(f)
    f.close()

    dir=f'./results_{args.dataset}/e{args.attack_epoch}_ld{args.lamda}_lr{args.lr}_mgraph_guided_c{args.apply_constraint}/b{args.budget_mode}_al{args.attack_loss}_s{args.save_opt}_{args.random_seed}/'
    f = open(dir + '/attack_feature_results.pkl', 'rb')
    df_2, df_anomaly_2, df_melted_2 = pickle.load(f)
    f.close()

    df_anomaly["attack types"] = "random"
    df_anomaly_2["attack types"] = "graph guided"
    df["attack types"] = "random"
    df_2["attack types"] = "graph guided"
    df_melted["attack types"] = "random"
    df_melted_2["attack types"] = "graph guided"

    df_anomaly_temp = pd.concat([df_anomaly, df_anomaly_2], ignore_index=True)
    df_temp = pd.concat([df, df_2], ignore_index=True)
    df_melted_temp = pd.concat([df_melted, df_melted_2], ignore_index=True)
    draw_figrues(df_anomaly_temp, df_temp, df_melted_temp, args,decimal)

def draw_analysis_figures(similarity_fun,analysis_data,decimal,args,relative_edges,total_edges):
    df = pd.DataFrame(analysis_data)
    df.head()
    print(df)
    #df['budget'] = [str(round(x * 100, decimal)) + "%" for x in df['budget']]
    #df['budget'] = [str(round(x * relative_edges/total_edges, decimal+5)) + "%" for x in df['budget']]
    if args.budget_mode != "attack_number":
        df['budget'] = ['{:.1e}%'.format(x * relative_edges / total_edges) for x in df['budget']]
    else:
        df['budget'] = [str(x) for x in df['budget']]
    plt.figure(constrained_layout=True)
    sns.barplot(x='budget', y='edge_modified', data=df,palette="Blues")
    plt.ylabel('edge modified')
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimal))
    plt.savefig(args.output_dir + f'AnalysisEdge_{similarity_fun}.pdf', dpi=300)
    plt.show()

    control_degrees = []
    budgets=[]
    for j in range(len(analysis_data['control_degrees'][0])):
        for i in range(len(analysis_data['budget'])):
            control_degrees.append(analysis_data['control_degrees'][i][j])
            budgets.append(analysis_data['budget'][i])
    df_degrees = pd.DataFrame(
        {'budget': budgets,
         'control_degrees': control_degrees})

    if args.budget_mode!="attack_number":
        df_degrees['budget'] = [str(round(x * 100, decimal)) + "%" for x in df_degrees['budget']]
    else:
        df_degrees['budget'] = [str(x) for x in df_degrees['budget']]
    plt.figure(constrained_layout=True)
    sns.boxplot(x='budget', y='control_degrees', data=df_degrees,palette="Greens")
    plt.ylabel('control node degrees')
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4e'))
    plt.savefig(args.output_dir + f'AnalysisDegree_{similarity_fun}.pdf', dpi=300)
    plt.show()

    f = open(args.output_dir + f'/attack_feature_analysis_{similarity_fun}.pkl', 'wb')
    pickle.dump([df, df_degrees], f)
    f.close()

if __name__ == '__main__':
    data_path = "../DataSets/NetworkIntrusion/KDD99/kddcup.data_10_percent"
    preprocessing_data(data_path,sample_size=10000)
