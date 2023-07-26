import numpy as np
from scipy.linalg import eigh
from utils import *
from embedding import *
from perturbation_attack import *
import random
import warnings
warnings.filterwarnings("ignore")
### This is adapted from "https://github.com/abojchevski/node_embedding_attack"
### (Adversarial Attacks on Node Embeddings via Graph Poisoning)

### Load and preprocess the data
graph = load_dataset('../data/cora.npz')
adj_matrix = graph['adj_matrix']
labels = graph['labels']

adj_matrix, labels = standardize(adj_matrix, labels)
n_nodes = adj_matrix.shape[0]

np.random.seed(0)
targets=np.array(range(1,20))

### Set hyperparameters


n_flips = 200
dim = 32
window_size = 5

### Generate candidate edge flips


#candidates = generate_candidates_removal(adj_matrix=adj_matrix)
candidates = generate_candidates_target_addition(adj_matrix,targets)
random.seed(0)
index=np.array(random.sample(range(0,candidates.shape[0]), min(10000,len(candidates))))
candidates2=candidates[index]
### Compute simple baselines


# b_eig_flips = baseline_eigencentrality_top_flips(adj_matrix, candidates, n_flips)
b_deg_flips = baseline_degree_top_flips(adj_matrix, candidates, n_flips, False)
b_rnd_flips = baseline_random_top_flips(candidates, n_flips, 0)


### Compute adversarial flips using eigenvalue perturbation
# our_flips = perturbation_top_flips(adj_matrix, candidates, n_flips, dim, window_size)
our_flips = target_perturbation_top_flips(adj_matrix, candidates2, n_flips, dim,targets,labels)

### Evaluate classification performance using the skipgram objective
for flips, name in zip([None, b_rnd_flips,b_deg_flips, our_flips],
                       ['cln', 'rnd', 'deg','our']):

    if flips is not None:
        adj_matrix_flipped = flip_candidates(adj_matrix, our_flips)
    else:
        adj_matrix_flipped = adj_matrix

    embedding = deepwalk_skipgram(adj_matrix_flipped, dim, window_size=window_size)
    f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, labels,targets)
    print('{}, F1: {:.2f} {:.2f}'.format(name, f1_scores_mean[0], f1_scores_mean[1]))


