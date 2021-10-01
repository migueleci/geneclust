# Clustering-based Function Prediction
## by Miguel Romero and Oscar Ramirez

 1. Source code

| File | Description |
| ---- | ----------- |
| spectral.py | Spectral clustering |
| gote.py | Gene Ontoloty Term Enrichment (GOte) |
| ancestors.py | Feature creations from p-values of GOte |
| data.py | Computation of hand-crafted properties of a network |
| featsel.py | Feature selections using SHAP |
| pipe.py | Supervised learning pipeline |

 2. Data

| File | Description |
| ---- | ----------- |
| gcn.csv | Gene co-expression network (edgelist) for Zea mays|
| affg.csv | Affinity graph (edgelist) for Zea mays |
| gfunc | Associations between genes and functions of level 1 in GO hierarchy |
