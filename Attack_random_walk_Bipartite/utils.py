import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import matplotlib.pyplot as plt
import json
import itertools
import os
import pickle
import pandas as pd
import csv
import warnings
from tqdm import tqdm
from sklearn.metrics import roc_curve,auc
import seaborn as sns;
sns.set()
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
plt.style.use('classic')
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['legend.title_fontsize'] = BIGGER_SIZE

def init_random_seed(SEED=2021):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
    warnings.filterwarnings("ignore")

'''load data'''
def Dataloader(datasets='',**kwargs):
    '''option of datasets=['RandomBipartite_ER','RandomBipartite_BA','AmazonReviews','AuthorPapers']'''
    '''generate bipartite graph based on binomial distribution(ER model)'''
    if datasets=='RandomBipartite_ER':
        k=kwargs['k']
        n=kwargs['n']
        p=kwargs['p']
        G=bipartite.generators.random_graph(k, n, p, seed=2021, directed=False)
    '''generate bipartite graph based on preferential attachment (BA model)'''
    if datasets=='RandomBipartite_BA':
        k=kwargs['k']
        p = kwargs['p']
        aseq=np.random.exponential(3, k)+3
        aseq=[int(d) for d in aseq]
        G=bipartite.preferential_attachment_graph(aseq, p, seed=2021)
        #G._node: get the attribute of bipartite set info
        n=len(G.nodes)-k
    if datasets == 'Software':
        number_records = kwargs['number_records']
        data = pd.read_csv('Datasets/Software.csv', header=None)  # Magazine_Subscriptions
        data.columns = ['products', 'users', 'rating', 'timestamp']
        # data=data.sort_values(by=['users'])
        edge_products = list(data[:number_records]['products'])
        edge_users = list(data[:number_records]['users'])
        prod_dict = dict([(y, x) for x, y in enumerate(sorted(set(edge_products)))])
        length_prod = len(prod_dict)
        user_dict = dict([(y, x + length_prod) for x, y in enumerate(sorted(set(edge_users)))])
        length_user = len(user_dict)
        prod_id = [prod_dict[x] for x in edge_products]
        user_id = [user_dict[x] for x in edge_users]
        print('range of product id:', min(prod_id), '~', max(prod_id))
        print('range of user id:', min(user_id), '~', max(user_id))
        # build graph
        G = nx.Graph()
        G.add_nodes_from(prod_id, bipartite=0)
        G.add_nodes_from(user_id, bipartite=1)
        G.add_edges_from(zip(prod_id, user_id))
        print('number of nodes in G:', len(G.nodes))
        k=length_prod
        n=length_user
    if datasets == 'Magzine':
        number_records = kwargs['number_records']
        data = pd.read_csv('Datasets/Magazine_Subscriptions.csv', header=None)  # Magazine_Subscriptions
        data.columns = ['products', 'users', 'rating', 'timestamp']
        # data=data.sort_values(by=['users'])
        edge_products = list(data[:number_records]['products'])
        edge_users = list(data[:number_records]['users'])
        prod_dict = dict([(y, x) for x, y in enumerate(sorted(set(edge_products)))])
        length_prod = len(prod_dict)
        user_dict = dict([(y, x + length_prod) for x, y in enumerate(sorted(set(edge_users)))])
        length_user = len(user_dict)
        prod_id = [prod_dict[x] for x in edge_products]
        user_id = [user_dict[x] for x in edge_users]
        print('range of product id:', min(prod_id), '~', max(prod_id))
        print('range of user id:', min(user_id), '~', max(user_id))
        # build graph
        G = nx.Graph()
        G.add_nodes_from(prod_id, bipartite=0)
        G.add_nodes_from(user_id, bipartite=1)
        G.add_edges_from(zip(prod_id, user_id))
        print('number of nodes in G:', len(G.nodes))
        k=length_prod
        n=length_user
    if datasets=='AuthorPapers':
        edge_author=[]
        edge_paper=[]
        number_records = kwargs['number_records']
        with open('Datasets/arxivData.json') as f:
            data = json.load(f)
            for i,d in enumerate(data[:number_records]):
                try:
                    authors=d['author'].replace('\'',"\"")
                    authors=json.loads(authors)
                    paper=d['id']
                    for a in authors:
                        if a['name'] !='.' and a['name'] !=':':
                            edge_author.append(a['name'])
                            edge_paper.append(paper)
                except:
                    continue
        paper_dict = dict([(y, x) for x, y in enumerate(set(edge_paper))])
        length_paper = len(paper_dict)
        author_dict = dict([(y, x + length_paper) for x, y in enumerate(set(edge_author))])
        length_author = len(author_dict)
        paper_id = [paper_dict[x] for x in edge_paper]
        author_id = [author_dict[x] for x in edge_author]
        print('range of paper id:', min(paper_id), '~', max(paper_id))
        print('range of author id:', min(author_id), '~', max(author_id))
        # build graph
        G = nx.Graph()
        G.add_nodes_from(paper_id, bipartite=0)
        G.add_nodes_from(author_id, bipartite=1)
        G.add_edges_from(zip(paper_id, author_id))
        print('number of nodes in G:', len(G.nodes))
        k=length_paper
        n=length_author
    return G,k,n


'''visualization'''
def visualization_graph(G,visualization_graph_flag,bipartite_layout=False):
    if visualization_graph_flag==True:
        top = [node for node in G._node.keys() if G._node[node]['bipartite'] == 0]
        bottom = [node for node in G._node.keys() if G._node[node]['bipartite'] == 1]
        subset_color = [
            "limegreen",
            "darkorange"
        ]
        color=[subset_color[i] for i in [0 if n in top else 1 for n in G.nodes]]
        options = {
            "node_size": 3,
            "edge_color": "black",
            "linewidths": 0,
            "width": 0.1,
        }
        if bipartite_layout == True:
            top = nx.bipartite.sets(G)[0]
            pos = nx.bipartite_layout(G, top)
            plt.figure(1, figsize=(15, 15))
            nx.drawing.nx_pylab.draw(G,node_color=color,pos=pos,**options)
            plt.title('Visualization of generated bipartite graph')
            #plt.savefig('Visualization of generated bipartite graph.jpg')
            plt.show()
        plt.figure(2, figsize=(15, 15))
        nx.drawing.nx_pylab.draw(G, node_color=color, **options)
        plt.title('Visualization of generated bipartite graph')
        #plt.savefig('Visualization of generated bipartite graph.jpg')
        plt.show()


def visualize_matrix(M,xlabel=None,ylabel=None):
    m,n=M.shape
    RGB_M = np.zeros((m,n, 3))
    M = M
    for i in range(m):
        for j in range(n):
            if M[i, j] == 1:
                RGB_M[i, j, :] = (0, 0, 1)#blue for original edges
            elif M[i, j] == 2:
                RGB_M[i, j, :] = (1, 0, 0)#red for injected edges
            else:
                RGB_M[i, j, :] = (1, 1, 1)#white
    plt.figure()
    plt.imshow(RGB_M)
    if xlabel!=None and ylabel!=None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.show()

def visualize_anomaly_scores(G,k,n,n0,bins_t,labels,All_mean_Scores,args):
    plt.figure(constrained_layout=True)
    plt.title('Distribution of original anomaly scores')
    plt.hist(1 - All_mean_Scores[0:n - n0], bins=50, facecolor="darkorange", edgecolor="black", alpha=0.3)
    plt.hist(1 - All_mean_Scores[n - n0:n], bins=bins_t, facecolor="red", edgecolor="black", alpha=0.4)
    anomaly = "red"
    normal = "darkorange"
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k", alpha=0.5) for c in [anomaly, normal]]
    labels_list = ["anomaly nodes", "normal nodes"]
    plt.legend(handles, labels_list, loc='upper left')
    plt.ylabel("frequency")
    plt.xlabel("anomaly scores")
    plt.tight_layout()
    plt.savefig(args.output_dir + 'BiGraphOriDistribution.pdf', dpi=300)
    # plt.show()

    Degree_bottoms = [d for node, d in G.degree if node >= k]

    plt.figure(constrained_layout=True)
    group = labels
    cdict = {0: "darkorange", 1: 'red'}
    ldict = {0: "normal nodes", 1: 'anomaly nodes'}
    fig, ax = plt.subplots()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(np.array(Degree_bottoms)[ix], 1 - All_mean_Scores[ix], c=cdict[g], label=ldict[g], s=120, alpha=0.5)
    ax.legend(loc="lower right")
    plt.title('Relation of anomaly scores and node degree')
    plt.ylabel("anomaly scores")
    plt.xlabel("node degree")
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(args.output_dir + 'BiGraphsimilarity_degrees.pdf', dpi=300)
    plt.show()


'''Anomaly injection'''
def injectCliqueCamo(G,k,n,k0,n0, p, type=1):
    '''Adapted from "FRAUDAR: Bounding Graph Fraud in the Face of Camouflage"'''
    # A=nx.adjacency_matrix(G).toarray()# this will sort the paper.
    # M=A[:k, k:]
    M, _=get_M_and_adj_matrix(G, k, n)
    M2 = np.zeros((k,n+n0))
    M2[:k,:n]=M
    rawSum = np.squeeze(M2.sum(axis = 1))
    colSumPart = rawSum[k0:k]
    colSumPartPro = np.int_(colSumPart)
    colIdx = np.arange(k0,k)
    population = np.repeat(colIdx, colSumPartPro, axis = 0)
    for i in range(k0):
        # inject clique
        for j in range(n,n+n0,1):
            if random.random() < p:
                M2[i,j] = 2
    for i in range(k0,k,1):
        # inject camo
        if type == 1:
            thres = p * k0 / (k - k0)
            for j in range(n,n+n0,1):
                if random.random() < thres:
                    M2[i,j] = 2
        if type == 2:
            thres = 2 * p * k0 / (k - k0)
            for j in range(n,n+n0,1):
                if random.random() < thres:
                    M2[i,j] = 2
        # biased camo
        if type == 3:
            colRplmt = random.sample(population, int(k0 * p))
            for j in range(n,n+n0,1):
                M2[colRplmt,j] = 2
    add_id=range(k+n,k+n+n0,1)
    G.add_nodes_from(add_id, bipartite=1)
    add_edges=0
    for i in range(k):
        for j in range(n,n+n0,1):
            if M2[i,j]==2:
                G.add_edge(i,k+j)
                add_edges=add_edges+1
    print('number of added edges in G:',add_edges )
    return G,M2


def reconstruct_graph(G,keep_nodes=''):
    top_id = [edge[0] for edge in G.edges]
    bottom_id = [edge[1] for edge in G.edges]
    keep_top_id = []
    keep_bottom_id = []
    for i in range(len(top_id)):
        if top_id[i] in keep_nodes and bottom_id[i] in keep_nodes:
            keep_top_id.append(top_id[i])
            keep_bottom_id.append(bottom_id[i])

    top_dict = dict([(y, x) for x, y in enumerate(sorted(set(keep_top_id)))])
    length_top = len(top_dict)
    bottom_dict = dict([(y, x + length_top) for x, y in enumerate(sorted(set(keep_bottom_id)))])
    length_bottom = len(bottom_dict)
    map_top_id = [top_dict[x] for x in keep_top_id]
    map_bottom_id = [bottom_dict[x] for x in keep_bottom_id]
    print('range of top id after reconstructed:', min(map_top_id), '~', max(map_top_id))
    print('range of bottom id after reconstructed:', min(map_bottom_id), '~', max(map_bottom_id))
    G = nx.Graph()
    G.add_nodes_from(map_top_id, bipartite=0)
    G.add_nodes_from(map_bottom_id, bipartite=1)
    #print(len(G.nodes))
    G.add_edges_from(zip(map_top_id, map_bottom_id))
    #G.edges
    return G,length_top,length_bottom

'''remove '''
def remove_low_degree(G,k,threshold=5,iteration=5):
    '''we need to remove it multiple times because
    removing current low degree nodes leads to more
    other low degree nodes '''

    for i in range(iteration):
        #bottom node
        remove_list=[]
        for node,d in G.degree:
            #print(node,d)
            if node>k and d<=threshold:#
                remove_list.append(node)
        G.remove_nodes_from(remove_list)

        #top node
        remove_list = []
        for node, d in G.degree:
            # print(node,d)
            if node < k and d < 1:  #
                remove_list.append(node)
        G.remove_nodes_from(remove_list)
    top = [node for node in G._node.keys() if G._node[node]['bipartite'] == 0]
    bottom = [node for node in G._node.keys() if G._node[node]['bipartite'] == 1]
    return G,len(top),len(bottom)


def numpy_to_torch_sparse(P):
    # P = coo_matrix(P)
    # values = P.data
    # indices = np.vstack((P.row, P.col))
    #
    # i = torch.LongTensor(indices)
    # v = torch.FloatTensor(values)
    # shape = P.shape
    # P = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    P=torch.tensor(P)
    P=P.to_sparse()
    return P

def get_M_and_adj_matrix(G,k,n):
    edges_list = G.edges
    M = np.zeros((k, n))
    adj_matrix = np.zeros((k + n, k + n))
    for e in edges_list:
        M[e[0], e[1] - k] = 1
        adj_matrix[e[0], e[1]] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return M, adj_matrix


def largest_indices(tensor, n):
    """Returns the n largest indices from a numpy array."""
    ary = tensor.detach().numpy()
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

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
    if title!=None:
        plt.title(f'Receiver operating curve ({title})', fontsize=20)
    else:
        plt.title(f'Receiver operating curve', fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(savedir, dpi=300)
    plt.show()
    plt.clf()


def detected(ranks,target_nodes,cutoff=0.01):
    total=len(ranks)
    anomaly_num=int(total*cutoff)
    detected_nodes=[n for n in target_nodes if ranks[n]<anomaly_num]
    detected_percentage=len(detected_nodes)/target_nodes.shape[0]
    return detected_percentage

def generate_candidates_addition(M, targets):
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
    k,n = M.shape[0],M.shape[1]
    K_nodes = [v for v in range(0,k)]
    candidates_list=[]
    for item in itertools.product(K_nodes,targets):
        if M[item[0],item[1]] ==0:
            candidates_list.append(item)
    candidates = np.array(candidates_list)
    return candidates

def baseline_random_top_flips(candidates, n_flips, seed):
    """Selects (n_flips) number of flips at random.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?, 2]
        The top edge flips from the candidate set
    """
    np.random.seed(seed)
    return candidates[np.random.permutation(len(candidates))[:n_flips]]


def baseline_degree_top_flips(adj_matrix, candidates, n_flips, complement):
    """Selects the top (n_flips) number of flips using degree centrality score of the edges.

    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param complement: bool
        Whether to look at the complement graph
    :return: np.ndarray, shape [?, 2]
        The top edge flips from the candidate set
    """
    if complement:
        adj_matrix = sp.csr_matrix(1-adj_matrix.toarray())
    deg = adj_matrix.sum(1)
    deg_argsort = (deg[candidates[:, 0]] + deg[candidates[:, 1]]).argsort()

    return candidates[deg_argsort[-n_flips:]]

def flip_candidates(adj_matrix, candidates):
    """Flip the edges in the candidate set to non-edges and vise-versa.

    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :return: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph with the flipped edges/non-edges.
    """
    adj_matrix_flipped = adj_matrix.copy()
    adj_matrix_flipped[candidates[:, 0], candidates[:, 1]] = 1 - adj_matrix[candidates[:, 0], candidates[:, 1]]
    adj_matrix_flipped[candidates[:, 1], candidates[:, 0]] = 1 - adj_matrix[candidates[:, 1], candidates[:, 0]]
    return adj_matrix_flipped

def draw_figures(data, args, combine=True):
    df = pd.DataFrame(data)
    df.head()
    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='maximum score', data=df, legend='full')
    plt.title('maximum score')
    if args.budget_mode=="totoal_edges":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=2))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.savefig(args.output_dir + 'BiGraphAttackResult_maximum_score.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='avarage score',data=df, legend='full')
    plt.title('average anomaly score')
    if args.budget_mode=="totoal_edges":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=2))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.savefig(args.output_dir + 'BiGraphAttackResult_avarage_score.pdf', dpi=300)
    plt.show()

    plt.figure(constrained_layout=True)
    sns.lineplot(x='budget', y='ranking',  data=df, legend='full')
    plt.title('Anomaly ranking')
    if args.budget_mode=="totoal_edges":
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=2))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=2))
    plt.savefig(args.output_dir + 'BiGraphAttackResult_ranking.pdf', dpi=300)
    plt.show()

    if combine==True:
        combined_df = pd.DataFrame({'detected rate': list(df["detected_top1%"])+list(df["detected_top5%"])+list(df["detected_top10%"]),'Detected rate':['in top 1%']*df.shape[0]+['in top 5%']*df.shape[0]+['in top 10%']*df.shape[0],'budget':list(df["budget"])+list(df["budget"])+list(df["budget"])})
        plt.figure(constrained_layout=True)
        sns.lineplot(x='budget', y='detected rate',hue="Detected rate", data=combined_df, legend='full', linewidth=2.0)
        plt.title('Detected rate')
        if args.budget_mode == "totoal_edges":
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        plt.savefig(args.output_dir + 'BiGraphAttackResult_detected_combined.pdf', dpi=300)
        plt.show()
    else:
        plt.figure(constrained_layout=True)
        sns.lineplot(x='budget', y='detected_top1%', data=df, legend='full', linewidth=2.0)
        plt.title('Detected rate')
        if args.budget_mode == "totoal_edges":
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        plt.savefig(args.output_dir + 'BiGraphAttackResult_detected_1.pdf', dpi=300)
        plt.show()

        plt.figure(constrained_layout=True)
        sns.lineplot(x='budget', y='detected_top5%', data=df, legend='full', linewidth=2.0)
        plt.title('Detected rate (in top 5%)')
        if args.budget_mode == "totoal_edges":
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        plt.savefig(args.output_dir + 'BiGraphAttackResult_detected_5.pdf', dpi=300)
        plt.show()

        plt.figure(constrained_layout=True)
        sns.lineplot(x='budget', y='detected_top10%', data=df, legend='full', linewidth=2.0)
        plt.title('Detected rate (in top 10%)')
        if args.budget_mode == "totoal_edges":
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        plt.savefig(args.output_dir + 'BiGraphAttackResult_detected_10.pdf', dpi=300)
        plt.show()

    print("results(figures) saved to:", args.output_dir)
    f = open(args.output_dir + '/attack_graph_results.pkl', 'wb')
    pickle.dump(df, f)
    f.close()