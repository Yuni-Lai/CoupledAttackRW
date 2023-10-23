# Coupled-space Attack against Random-Walk-based Anomaly Detection
This is the source code for "Coupled-Space Attacks against Random-Walk-based Anomaly Detection".  

## Structure 
```bash
.
├── Attack_random_walk_Bipartite
│   ├── Datasets
│   │   ├── Magazine_Subscriptions.csv
│   │   ├── __init__.py
│   │   ├── arxivData.json
│   │   └── paperCrawl.py
│   ├── DeepWalkAttack
│   │   ├── node_embedding_attack
│   │   │   ├── embedding.py
│   │   │   ├── main.py
│   │   │   ├── perturbation_attack.py
│   │   │   └── utils.py
│   │   └── setup.py
│   ├── __init__.py
│   ├── draw.py
│   ├── main.py
│   ├── models.py
│   ├── readme.md
│   ├── run.sh
│   └── utils.py
├── Attack_random_walk_Proximity
│   ├── 01_attack_graph
│   │   ├── DeepWalkAttack
│   │   │   ├── README.md
│   │   │   ├── node_embedding_attack
│   │   │   │   ├── __init__.py
│   │   │   │   ├── embedding.py
│   │   │   │   ├── main.py
│   │   │   │   ├── perturbation_attack.py
│   │   │   │   └── utils.py
│   │   │   └── setup.py
│   │   ├── baselines.py
│   │   ├── draw.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── run.sh
│   │   └── utils.py
│   ├── 02_attack_feature
│   │   ├── draw.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── run.sh
│   │   └── utils.py
│   └── DataSets
│       ├── Mnist
│       │   └── mnist.mat
│       └── NetworkIntrusion
│           └── KDD99
│               ├── data.pkl
│               ├── data_constraint_info.pkl
│               └── kddcup.data_10_percent.gz
├── README.md
└── requirements
    ├── py37.txt
    └── py37.yml
```
## Environment Setup
Both of the two files are required:
```bash
conda env create -f requirements/py37.yml
conda activate py37
pip install -r requirements/py37.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yaml or .txt file and delete that line.

## Dataset  
### KDD-99:
Please click the link below to download the "kddcup.data_10_percent.zip A 10% subset. (2.1M; 75M Uncompressed)" dataset:  
https://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data  
The data preprocessing in main.py will generate KDD99/data.pkl and data_constraint_info.pkl files. 
### MNIST:
Please click the link below to download the "mnist.mat":   
https://odds.cs.stonybrook.edu/mnist-dataset/

### Author-Paper:  This dataset is the papers crawled from arXiv preprint database.  
The crawler is from the repository: [GabrielePisciottaarxiv-dataset-download](https://github.com/GabrielePisciotta/arxiv-dataset-download). There are many subjects of papers, such as cs.AI, math.ML.
Nodes U are papers, and nodes V are authors, edges < u, v > means the author v shown in the paper u. Please see our paperCrawl.py and Attack_random_walk_Bipartite/main.py for detailed data processing. 

### Amazon Reviews:  The [Amason Reviews Data (2018)](https://nijianmo.github.io/amazon/index.html).   
This data includes different kinds of products and their ratings from users. 
Nodes U are products, and nodes V are users, edges < u, v > means the user v gives a review to the product u.
  
## Our attack:
Attack_random_walk_Proximity (InDi-RWAD):  
01_attack_graph: graph-space attack.  
02_attack_graph: feature-space attack.  

Attack_random_walk_Bipartite (Di-RWAD):  
graph-space attack only.

Run our proposed attack
examples:
```bash
python main.py -dataset "KDD99" -attack_mode 'closed-form' -gpuID 0
python main.py -dataset "KDD99" -attack_mode 'alternative' -gpuID 0
```
or 
```bash
bash run.sh
```
Run the baselines:

examples:
```bash
python -u main.py -dataset "KDD99" -attack_mode 'DeepWalk' -gpuID 0
python -u main.py -dataset "KDD99" -attack_mode 'random' -gpuID 0
```


```bash
@misc{lai2023dualspace,
  title = {Coupled-Space Attacks against Random-Walk-based Anomaly Detection},
  author = {Yuni, Lai and Marcin, Waniek and Liying, Li and Jingwen, Wu and Yulin, Zhu
            and Tomasz P., Michalak and Talal, Rahwan and Kai, Zhou},
  year = {2023},
  journal = {arXiv preprint:2307.14387},
}
```
