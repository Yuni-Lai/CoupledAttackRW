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

### This is adapted from "https://github.com/abojchevski/node_embedding_attack"
### (Adversarial Attacks on Node Embeddings via Graph Poisoning)

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


def baseline_eigencentrality_top_flips( adj_matrix, candidates, n_flips):
    """Selects the top (n_flips) number of flips using eigencentrality score of the edges.
    Applicable only when removing edges.

    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :return: np.ndarray, shape [?, 2]
        The top edge flips from the candidate set
    """
    edges = np.column_stack(sp.triu(adj_matrix, 1).nonzero())
    line_graph = construct_line_graph(adj_matrix)
    eigcentrality_scores = nx.eigenvector_centrality_numpy(nx.Graph(line_graph))
    eigcentrality_scores = {tuple(edges[k]): eigcentrality_scores[k] for k, v in eigcentrality_scores.items()}
    eigcentrality_scores = np.array([eigcentrality_scores[tuple(cnd)] for cnd in candidates])
    scores_argsrt = eigcentrality_scores.argsort()
    return candidates[scores_argsrt[-n_flips:]]


def baseline_degree_top_flips( adj_matrix, candidates, n_flips, complement):
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
        adj_matrix = sp.csr_matrix(1 - adj_matrix.toarray())
    deg = adj_matrix.sum(1)
    deg_argsort = (deg[candidates[:, 0]] + deg[candidates[:, 1]]).argsort()
    ### This is adapted from "https://github.com/abojchevski/node_embedding_attack"
    ### (Adversarial Attacks on Node Embeddings via Graph Poisoning)
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