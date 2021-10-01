# Clustering-based Function Prediction
## by Miguel Romero and Oscar Ramirez

 1. Source code

| File | Description |
| ---- | ----------- |
| spectral.py | Spectral clustering |
| ancestors.py | Feature creations from p-values obtained from Gene Ontoloty Term Enrichment |
| data.py | Computation of hand-crafted properties of a network |
| featsel.py | Feature selections using SHAP |
| pipe.py | Supervised learning pipeline |
| plots.py | Module for plotting |

 2. Data

| File | Description |
| ---- | ----------- |
| genes.txt | List of genes on gene co-expression network for Zea mays |
| gcn.csv | Gene co-expression network (edgelist) for Zea mays [1] |
| affg.csv | Affinity graph (edgelist) for Zea mays [1] |
| gfunc | Associations between genes and functions of level 1 in GO hierarchy |
| go-basic.obo | Gene Ontology hierarchy database |

[1]: https://drive.google.com/drive/folders/1aIahl4a75BgicCcybpZAT2UPzdyZJE3T?usp=sharing
