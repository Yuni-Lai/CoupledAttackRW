# Dual-space Attack against Random-Walk-based Anomaly Detection
This is the source code for "Dual-Space Attacks against Random-Walk-based Anomaly Detection".  

This repo contains Dual-space attacks against RWAD over indirectly accessible graphs (InDi-RWAD).   
Specifically, for RWAD based on Proximity Graph. 

Attack Di-RWAD (Bipartite graph):  
For attacks on RWAD over directly accessible graph (Di-RWAD), please go to another anonymized GitHub page:  
https://anonymous.4open.science/r/Attack_random_walk_Bipartite

## Structure 
```bash
.
├── 01_attack_graph
│   ├── DeepWalkAttack
│   │   ├── node_embedding_attack
│   │   │   ├── __init__.py
│   │   │   ├── embedding.py
│   │   │   ├── main.py
│   │   │   ├── perturbation_attack.py
│   │   │   └── utils.py
│   │   └── setup.py
│   ├── baselines.py
│   ├── draw.py
│   ├── main.py
│   ├── model.py
│   ├── run.sh
│   └── utils.py
├── 02_attack_feature
│   ├── draw.py
│   ├── main.py
│   ├── model.py
│   ├── run.sh
│   └── utils.py
├── 03_transfer_attack
│   ├── main.py
│   ├── result.py
│   └── utils.py
├── DataSets
│   ├── Mnist
│   │   └── mnist.mat
│   ├── NetworkIntrusion
│   │   └── KDD99
│   │       ├── data.pkl
│   │       ├── data_constraint_info.pkl
│   │       ├── kddcup.data_10_percent
│   │       └── kddcup.data_10_percent.gz
├── README.md
└── requirements
    ├── py37.txt
    └── py37.yml
```
## Environment

```bash
conda env create -f py37.yml
conda activate py37
pip install -r py37.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yaml or .txt file and delete that line.

## Dataset  
### KDD-99:
Please click the link below to download the "kddcup.data_10_percent.zip A 10% subset. (2.1M; 75M Uncompressed)" dataset:  
https://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data
The data preprocessing in main.py will generate KDD99/data.pkl file. 
### MNIST:
Please click the link below to download the "mnist.mat": 
https://odds.cs.stonybrook.edu/mnist-dataset/

## Run our attack:
01_attack_graph: graph space attack.  
02_attack_graph: feature space attack.  
03_transfer_attack: feature space transfer attack.  

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


