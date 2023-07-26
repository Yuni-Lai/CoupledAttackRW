# Attack random walk based anomaly detection （Bipartite）

## Introduction
Random walk is widely used in graph anomaly detection.

```
├── Datasets
│   ├── Magazine_Subscriptions.csv
│   ├── arxivData.json
│   └── paperCrawl.py
├── draft.py
├── main.py
├── models.py
├── readme.md
├── test.py
└── utils.py
```
---
### Datasets
- **Author-Paper:**  This dataset is the papers crawled from arxiv preprint database. The crawler is from the repository: [GabrielePisciotta
/
arxiv-dataset-download](https://github.com/GabrielePisciotta/arxiv-dataset-download). There are many subjects of papers we can choose to obtain, such as cs.AI, math.ML.
Nodes U are papers, and nodes V are authors, edges < u, v > means the author v shown in the paper u.

- **Amazon Reviews:**  The [Amason Reviews Data (2018)](https://nijianmo.github.io/amazon/index.html). This data including different kind of products and their ratings from users. 
Nodes U are products, and nodes V are users, edges < u, v > means the user v gives review to the product u.
  
- **Generated BA Bipartite Graph:** Networkx bipartite generators: [preferential attachment graph](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.generators.preferential_attachment_graph.html#networkx.algorithms.bipartite.generators.preferential_attachment_graph).
  
- **Generated ER Bipartite Graph:** Erd¨os-R´enyi (ER) random graph model. Networkx bipartite generator: [random graph](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.generators.random_graph.html#networkx.algorithms.bipartite.generators.random_graph).
  This is a bipartite version of the binomial (Erdős-Rényi) graph. The graph is composed of two partitions. Set A has nodes 0 to (n - 1) and set B has nodes n to (n + m - 1).
  
### Environment setup
```
$ conda env create -f /requirements/py38.yaml
$ conda activate Python38
$ pip install -r /requirements/py38.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yaml or txt file and delete the line. 



