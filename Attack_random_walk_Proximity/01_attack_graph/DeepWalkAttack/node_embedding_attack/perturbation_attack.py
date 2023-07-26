"""
Implementation of the method proposed in the paper:

'Adversarial Attacks on Node Embeddings via Graph Poisoning'
Aleksandar Bojchevski and Stephan Günnemann, ICML 2019
http://proceedings.mlr.press/v97/bojchevski19a.html

Copyright (C) owned by the authors, 2019
"""

import numba
import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl
# import tensorflow as tf
import networkx as nx
#from node_embedding_attack.utils import *
from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score,recall_score
from sklearn.model_selection import StratifiedShuffleSplit

from joblib import Memory

mem = Memory(cachedir='/tmp/joblib')

#mem.clear(warn=True)

def perturbation_top_flips(adj_matrix, candidates, n_flips, dim, window_size):
    """Selects the top (n_flips) number of flips using our perturbation attack.

    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param dim: int
        Dimensionality of the embeddings.
    :param window_size: int
        Co-occurence window size.
    :return: np.ndarray, shape [?, 2]
        The top edge flips from the candidate set
    """
    n_nodes = adj_matrix.shape[0]
    # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    delta_w = 1 - 2 * adj_matrix[candidates[:, 0], candidates[:, 1]].A1

    # generalized eigenvalues/eigenvectors
    deg_matrix = np.diag(adj_matrix.sum(1).A1)
    vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), deg_matrix)

    loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes, dim, window_size)
    top_flips = candidates[loss_for_candidates.argsort()[-n_flips:]]

    return top_flips

def sumpowers1(x, power):
    n = x.shape[0]
    sum_powers = np.zeros((power,n))
    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)
    return sum_powers.sum(0)

def sumpowers2(x, power):
    n,k = x.shape
    sum_powers = np.zeros((power,n,k))
    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)
    return sum_powers.sum(0)


#@mem.cache
def target_perturbation_top_flips(adj_matrix, candidates, n_flips, dim,targets,labels,save_dir=None):
    """Selects the top (n_flips) number of flips using targeted perturbation attack. For 2 classes classification!!!!
    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param dim: int
        Dimensionality of the embeddings.
    :return: np.ndarray, shape [?, 2]
        The top edge flips from the candidate set
    """
    n_nodes = adj_matrix.shape[0]
    n_candidates=len(candidates)
    # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    delta_w = 1 - 2 * adj_matrix[candidates[:, 0], candidates[:, 1]].A1
    # generalized eigenvalues/eigenvectors
    deg_matrix = np.diag(adj_matrix.sum(1).A1)
    vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), deg_matrix)# the eig values are already sorted in acsending order
    vals_org=vals_org[::-1]
    vecs_org=vecs_org[::-1]

    embedding_org = np.einsum('ij,ii->ji', np.transpose(vecs_org[:,:dim]), np.diag(sumpowers1(vals_org, 5))[:dim,:dim])
    delta_eigvals = estimate_delta_eigvals(candidates, adj_matrix, vals_org, vecs_org)
    degrees=np.squeeze(np.array(np.sum(adj_matrix, axis=0)))
    pinvs=get_pinvs(adj_matrix, vals_org, dim)
    delta_eigvecs = estimate_delta_eigenvecs(candidates, delta_w, degrees, vals_org, vecs_org, delta_eigvals, pinvs, dim)
    vecs_new = np.tile(np.transpose(vecs_org[:,:dim]), (n_candidates,1,1))+delta_eigvecs
    vals_new = np.tile(vals_org,(n_candidates,1))+delta_eigvals
    embedding_target_new = estimate_target_embeding(vals_new[:,:dim], vecs_new,n_candidates,targets)
    lr,embedding_target_new_normalized=classifier(embedding_org,embedding_target_new,labels,targets)
    candidate_margin=lr.predict_proba(embedding_target_new_normalized)[:,0].reshape(n_candidates,len(targets))# For 2 classes classification!!!!
    candidate_margin_mean=np.mean(candidate_margin,axis=1)
    top_flips = candidates[candidate_margin_mean.argsort()[-n_flips:]]
    if save_dir!=None:
        f = open(save_dir, 'wb')
        pickle.dump([candidate_margin_mean], f)
        f.close()
    return top_flips

#@numba.jit(nopython=True)
def estimate_target_embeding(vals_new,vecs_new,n_candidates,target):
    vals_new_power = sumpowers2(vals_new, 5)
    vals_new_diag = np.einsum('ij,jk->ijk', vals_new_power, np.eye(vals_new_power.shape[1], dtype=vals_new_power.dtype))
    embedding_new = np.einsum('ijk,ijl->ikl',vecs_new[:,:,target], vals_new_diag)
    return embedding_new

def classifier(embedding_org,embedding_target_new,labels,targets):
    embedding_all = normalize(np.concatenate((embedding_org, embedding_target_new.reshape(-1,32)), axis=0))
    embedding_org_normalized = embedding_all[:embedding_org.shape[0],:]
    embedding_target_new_normalized = embedding_all[embedding_org.shape[0]:,:]
    train_ratio=0.8
    seed=0
    labels=np.squeeze(labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed)
    split_train, split_test = next(sss.split(embedding_org_normalized, labels))
    features_train = embedding_org_normalized[split_train]
    features_test = embedding_org_normalized[split_test]
    labels_train = labels[split_train]
    labels_test = labels[split_test]
    features_target = embedding_org_normalized[targets]
    labels_target = labels[targets]
    lr = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto',class_weight='balanced')
    lr.fit(features_train, labels_train)
    test_predict = lr.predict(features_test)
    recall=recall_score(labels_test, test_predict)
    f1_micro = f1_score(labels_test, test_predict, average='micro')
    f1_macro = f1_score(labels_test, test_predict, average='macro')
    print(f"testing set:f1_micro:{f1_micro},f1_macro:{f1_macro},recall:{recall}")
    target_predict = lr.predict(features_target)
    recall = recall_score(labels_target, target_predict)
    f1_micro = f1_score(labels_target, target_predict, average='micro')
    f1_macro = f1_score(labels_target, target_predict, average='macro')
    print(f"target set:f1_micro:{f1_micro},f1_macro:{f1_macro},recall:{recall}")
    # lr.predict_proba(features_target)
    return lr,embedding_target_new_normalized


@numba.jit(nopython=True)
def estimate_loss_with_delta_eigenvals(candidates, flip_indicator, vals_org, vecs_org, n_nodes, dim, window_size):
    """Computes the estimated loss using the change in the eigenvalues for every candidate edge flip.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips,
    :param flip_indicator: np.ndarray, shape [?]
        Vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :param n_nodes: int
        Number of nodes
    :param dim: int
        Embedding dimension
    :param window_size: int
        Size of the window
    :return: np.ndarray, shape [?]
        Estimated loss for each candidate flip
    """

    loss_est = np.zeros(len(candidates))
    for x in range(len(candidates)):
        i, j = candidates[x]
        vals_est = vals_org + flip_indicator[x] * (
                2 * vecs_org[i] * vecs_org[j] - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2))

        vals_sum_powers = sum_of_powers(vals_est, window_size)

        loss_ij = np.sqrt(np.sum(np.sort(vals_sum_powers ** 2)[:n_nodes - dim]))
        loss_est[x] = loss_ij

    return loss_est


#@numba.jit(nopython=True)
#@mem.cache
def estimate_delta_eigenvecs(candidates, flip_indicator, degrees, vals_org, vecs_org, delta_eigvals, pinvs, dim):
    """Computes the estimated change in the eigenvectors for every candidate edge flip.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips,
    :param flip_indicator: np.ndarray, shape [?]
        Vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    :param degrees: np.ndarray, shape [n]
        Vector of node degrees.
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :param delta_eigvals: np.ndarray, shape [?, n]
        Estimated change in the eigenvalues for all candidate edge flips
    :param pinvs: np.ndarray, shape [k, n, n]
        Precomputed pseudo-inverse matrices for every dimension
    :return: np.ndarray, shape [?, n, k]
        Estimated change in the eigenvectors for all candidate edge flips
    """
    n_nodes = vecs_org.shape[0]
    n_candidates = len(candidates)
    delta_eigvecs = np.zeros((n_candidates, dim, n_nodes))

    for k in range(dim):
        cur_eigvecs = vecs_org[:, k]
        cur_eigvals = vals_org[k]
        # for c in range(n_candidates):
        #     degree_eigvec1 = (-delta_eigvals[c, k] * degrees) * cur_eigvecs
        #     i, j = candidates[c]
        #     degree_eigvec1[i] += cur_eigvecs[j] - cur_eigvals * cur_eigvecs[i]
        #     degree_eigvec1[j] += cur_eigvecs[i] - cur_eigvals * cur_eigvecs[j]
        #     delta_eigvecs1[c, k] = np.dot(pinvs[k], flip_indicator[c] * degree_eigvec1)
        degree_eigvec = (-np.expand_dims(delta_eigvals[:, k], axis=1) * np.tile(degrees, (n_candidates,1))) * np.tile(cur_eigvecs, (n_candidates,1))
        degree_eigvec[range(0,n_candidates),candidates[:,0]] += np.squeeze(np.expand_dims(cur_eigvecs[candidates[:,1]], axis=1)- cur_eigvals * np.expand_dims(cur_eigvecs[candidates[:,0]], axis=1))
        degree_eigvec[range(0,n_candidates),candidates[:, 1]] += np.squeeze(np.expand_dims(cur_eigvecs[candidates[:, 0]],
                                                          axis=1) - cur_eigvals * np.expand_dims(
            cur_eigvecs[candidates[:, 1]], axis=1))
        delta_eigvecs[:, k] =  np.transpose(np.dot(pinvs[k],np.transpose(degree_eigvec)))
    return delta_eigvecs
    ### the code above return exactly the same the folows:###
    # delta_eigvecs1 = np.zeros((n_candidates, dim, n_nodes))
    # for k in range(dim):
    #     cur_eigvecs = vecs_org[:, k]
    #     cur_eigvals = vals_org[k]
    #     for c in range(n_candidates):
    #         degree_eigvec1 = (-delta_eigvals[c, k] * degrees) * cur_eigvecs
    #         i, j = candidates[c]
    #         degree_eigvec1[i] += cur_eigvecs[j] - cur_eigvals * cur_eigvecs[i]
    #         degree_eigvec1[j] += cur_eigvecs[i] - cur_eigvals * cur_eigvecs[j]
    #         delta_eigvecs1[c, k] = np.dot(pinvs[k], flip_indicator[c] * degree_eigvec1)
    # return delta_eigvecs1



def estimate_delta_eigvals(candidates, adj_matrix, vals_org, vecs_org):
    """Computes the estimated change in the eigenvalues for every candidate edge flip.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :return: np.ndarray, shape [?, n]
        Estimated change in the eigenvalues for all candidate edge flips
    """
    # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    delta_w = 1 - 2 * adj_matrix[candidates[:, 0], candidates[:, 1]].A1

    delta_eigvals = delta_w[:, None] * (2 * vecs_org[candidates[:, 0]] * vecs_org[candidates[:, 1]]
                                        - vals_org * (
                                                vecs_org[candidates[:, 0]] ** 2 + vecs_org[candidates[:, 1]] ** 2))

    return delta_eigvals


@mem.cache
def get_pinvs(adj_matrix, vals_org, dim):
    """ Precomputes the pseudo-inverse matrices for every dimension.

    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param dim: int
        Embedding dimension
    :return:  np.ndarray, shape [k, n, n]
        Pseudo-inverse matrices for every dimension
    """
    deg_matrix = sp.diags(adj_matrix.sum(0).A1)
    pinvs = []
    for k in range(100):
        print(k)
        try:
            # pinvs.append(-np.linalg.pinv((adj_matrix - vals_org[k] * deg_matrix).toarray()))
            pinvs.append(-np.linalg.inv((adj_matrix - vals_org[k] * deg_matrix).toarray()))
        except:
            print('error')
            pass
        # except np.linalg.LinAlgError:
        #     print('error')
        #     pinvs.append(-spl.pinv((adj_matrix - vals_org[k+offset] * deg_matrix).toarray()))
        if len(pinvs)==dim:
            return np.stack(pinvs)
    # sum_of_powers = transition_matrix
    # last = transition_matrix
    # for i in range(1, pow):
    #     last = last.dot(transition_matrix)
    #     sum_of_powers += last


def estimate_loss_with_perturbation_gradient(candidates, adj_matrix, n_nodes, window_size, dim, num_neg_samples):
    """Computes the estimated loss using the gradient defined with eigenvalue perturbation.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param n_nodes: int
        Number of nodes in the graph
    :param window_size: int
        Size of the window
    :param dim: int
        Size of the embedding
    :param num_neg_samples: int
        Number of negative samples
    :return:
    """
    adj_matrix_tf, logM_tf, eigenvecs_tf, loss, adj_matrix_grad_tf = _get_gradient_estimator(
        n_nodes, window_size, dim, num_neg_samples)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    logM = sess.run(logM_tf, {adj_matrix_tf: adj_matrix.toarray()})
    logM = sp.csr_matrix(logM)
    eigenvals, eigenvecs = sp.linalg.eigsh(logM, dim)

    adj_matrix_grad = sess.run(adj_matrix_grad_tf, {adj_matrix_grad_tf: adj_matrix.toarray(), eigenvecs_tf: eigenvecs})[
        0]
    sig_est_grad = adj_matrix_grad[candidates[:, 0], candidates[:, 1]] + adj_matrix_grad[
        candidates[:, 1], candidates[:, 0]]
    ignore = sig_est_grad < 0
    sig_est_grad[ignore] = - 1

    return sig_est_grad

def _get_gradient_estimator(n_nodes, window_size, dim, num_neg_samples):
    """Define a tensorflow computation graph used to estimate the loss using the perturbation gradient.

    :param n_nodes: int
        Number of nodes in the graph
    :param window_size: int
        Size of the window
    :param dim: int
        Size of the embedding
    :param num_neg_samples: int
        Number of negative samples
    :return: (tf.placeholder, ...)
        Tensorflow placeholders used to estimate the loss.
    """
    adj_matrix = tf.placeholder(tf.float64, shape=[n_nodes, n_nodes])

    deg = tf.reduce_sum(adj_matrix, 1)
    volume = tf.reduce_sum(adj_matrix)

    transition_matrix = adj_matrix / deg[:, None]

    sum_of_powers = transition_matrix
    last = transition_matrix
    for i in range(1, window_size):
        last = tf.matmul(last, transition_matrix)
        sum_of_powers += last

    M = sum_of_powers / deg * volume / (num_neg_samples * window_size)
    logM = tf.log(tf.maximum(M, 1.0))

    norm_logM = tf.square(tf.norm(logM, ord=2))

    eigenvecs = tf.placeholder(tf.float64, shape=[n_nodes, dim])
    eigen_vals = tf.reduce_sum(eigenvecs * tf.matmul(logM, eigenvecs), 0)
    loss = tf.sqrt(norm_logM - tf.reduce_sum(tf.square(eigen_vals)))

    adj_matrix_grad = tf.gradients(loss, adj_matrix)

    return adj_matrix, logM, eigenvecs, loss, adj_matrix_grad

def baseline_eigencentrality_top_flips(adj_matrix, candidates, n_flips):
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

def add_by_remove(adj_matrix, candidates, n_flips, dim, window_size, c_rnd, seed=0):
    """
    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param dim: int
         Embedding dimension
     :param window_size: int
        Size of the window
    :param c_rnd: int
        Multiplicative constant for the number of other candidates to randomly select.
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?, 2]
        The top edge flips from the candidate set
    """
    np.random.seed(seed)

    n_nodes = adj_matrix.shape[0]

    rnd_perm = np.random.permutation(len(candidates))[:c_rnd * n_flips]
    candidates_add = candidates[rnd_perm]
    assert len(candidates_add) == c_rnd * n_flips

    adj_matrix_add = flip_candidates(adj_matrix, candidates_add)

    vals_org_add, vecs_org_add = spl.eigh(adj_matrix_add.toarray(), np.diag(adj_matrix_add.sum(1).A1))
    flip_indicator = 1 - 2 * adj_matrix_add[candidates[:, 0], candidates[:, 1]].A1

    loss_est = estimate_loss_with_delta_eigenvals(candidates_add, flip_indicator,
                                                  vals_org_add, vecs_org_add, n_nodes, dim, window_size)

    loss_argsort = loss_est.argsort()

    top_candidates = candidates_add[loss_argsort[:n_flips]]

    assert len(top_candidates) == n_flips

    return top_candidates

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
    try:
        deg = adj_matrix.sum(1).A1
    except:
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