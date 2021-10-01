# Clustering-based Function Prediction
## by Miguel Romero and Oscar Ramirez

This repository contains the source code and data used for the paper:

    Supervised Gene Function Prediction using Spectral Clustering on Gene Co-expression Networks
    Miguel Romero, Óscar Ramírez, Jorge Finke, and Camilo Rocha
    Accepted in Complex Networks 2021


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
| gcn.csv | Gene co-expression network (edgelist) for Zea mays (available in [link]) |
| affg.csv | Affinity graph (edgelist) for Zea mays (available in [link]) |
| gfunc | Associations between genes and functions of level 1 in GO hierarchy |
| go-basic.obo | Gene Ontology hierarchy database |

[link]: https://drive.google.com/drive/folders/1aIahl4a75BgicCcybpZAT2UPzdyZJE3T?usp=sharing
