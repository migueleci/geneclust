#!/usr/bin/env python3
# coding: utf-8

# Gene function prediction - Data preprocessing
# Miguel Romero - Oscar Ramirez, jul 2nd

import os
import sys
import json
import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx
from tqdm import tqdm
from time import time
from matplotlib import pyplot as plt

# Node embedding
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering

# Cross-validation and scaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)


##################################
# 1. Load data and create matrices
##################################

path = '/users/grupofinke/ramirez/xgb' #'/home/miguel/projects/omics/maize_data/xgb'
data_gcn = pd.read_csv('{0}/edgelist.csv'.format(path))
data_term_def = pd.read_csv('{0}/countFunctions.csv'.format(path))
data_functions = pd.read_csv('{0}/l1func.csv'.format(path))

data_gcn = data_gcn[['source', 'target']]
data_term_def = data_term_def[['Function', 'Description']]

def array2list(str):
  str = str.replace("\'",'\"')
  return json.loads(str)
data_functions['funcs'] = data_functions.funcs.apply(lambda t: array2list(t))

data = list()
for x in data_functions.itertuples(index=False):
  for f in x[1]:
    data.append((x[0],f))

data_gene_term = pd.DataFrame(data, columns=["gene","term"])
data_gene_term = data_gene_term.astype({'gene':int,'term':str})
data_gene_term.to_csv('functions.csv', index=False)

G = np.array(sorted(list(set(data_gcn['source'].tolist()+data_gcn['target'].tolist()))))
nG = len(G)

F = np.array(sorted(list(set(data_term_def['Function'].tolist()+data_gene_term['term'].tolist()))))
nF, idxF = len(F), dict([(g,i) for i,g in enumerate(F)])

print('**Initial data**')
print('Genes: \t\t{0}'.format(len(G)))
print('Genes co-expressed: \t{0:.0f}'.format(len(data_gcn)))
print('Functions: \t{0}'.format(len(data_term_def)))
print('Genes annoted \t{0}'.format(len(data_gene_term['gene'].unique())))
print('Gene-functions assoc.: \t{0}'.format(len(data_gene_term)))


# GCN matrix
# nG:number of genes, idxP:gene index map

gcn = np.zeros((nG,nG))
for edge in tqdm([tuple(x) for x in data_gcn.to_numpy()]):
  u, v = edge[0], edge[1]
  gcn[u][v] = gcn[v][u] = 1

# gene by go matrix
gene_by_go = np.zeros((nG,nF))
for edge in tqdm([tuple(x) for x in data_gene_term.to_numpy()]):
  u, v = edge[0], idxF[edge[1]]
  gene_by_go[u,v] = 1



#####################################
# 2. Prepare term data for prediction
#####################################

print()

# list sub-hierarchies
df_terms = pd.DataFrame(columns=['Term', 'Genes'])
for i in range(nF):
  data = [F[i], np.count_nonzero(gene_by_go[:,i])]
  df_terms.loc[i] = data

print(df_terms)



###################
# 3. Dataset design
###################

# Scale data
def scale_data(data):
  # MinMaxScaler does not modify the distribution of data
  minmax_scaler = MinMaxScaler() # Must be first option
  rob_scaler = RobustScaler() # RobustScaler is less prone to outliers

  new_data = pd.DataFrame()
  for fn in data.columns:
    scaled_feature = minmax_scaler.fit_transform(data[fn].values.reshape(-1,1))
    new_data[fn] = scaled_feature[:,0].tolist()

  return new_data

# compute graph properties and feature embedding for each sub-hierarchy
s = time()
terms_pred_idx = np.arange(nF)
genes_pred_idx = np.arange(nG)

# create sub matrix terms_hier_idx hierarchy
sm_gcn = gcn
sm_gene_by_go = gene_by_go

# create graph for gcn (hierarchy gcn)
sm_gcn_nx = nx.Graph()
sm_gcn_edgelist = np.transpose(np.nonzero(sm_gcn)).tolist()
sm_gcn_nx.add_nodes_from(np.arange(len(genes_pred_idx)))
sm_gcn_nx.add_edges_from(sm_gcn_edgelist)
print(nx.info(sm_gcn_nx))



# node embedding for prediction
dimensions = 16
p, q = 1, 0.5

node2vec = Node2Vec(sm_gcn_nx, dimensions=dimensions, walk_length=5, num_walks=300, workers=64, p=p, q=q)
model = node2vec.fit(window=5, min_count=1, batch_words=5, workers=64)
embeddings = np.array([model.wv.get_vector(str(x)) for x in list(sm_gcn_nx.nodes)])

# dimensionality reduction for clustering
tsne = TSNE(n_components=2, random_state=7, perplexity=15)
embeddings_2d = tsne.fit_transform(embeddings)

clustering_model = AffinityPropagation(damping=0.9)
clustering_model.fit(embeddings_2d)
yhat = clustering_model.predict(embeddings_2d)

# dimensionality reduction for small representation
n_components = 1
tsne = TSNE(n_components=n_components, random_state=7, perplexity=15)
embeddings_1d = tsne.fit_transform(embeddings)



# igraph
sm_gcn_nx = ig.Graph.Adjacency((sm_gcn > 0).tolist())
sm_gcn_nx.to_undirected()

# get node properties form graph
clust = np.array(sm_gcn_nx.transitivity_local_undirected(mode="zero"))
deg = np.array(sm_gcn_nx.degree())
neigh_deg = np.array(sm_gcn_nx.knn()[0])
centr_betw = np.array(sm_gcn_nx.betweenness(directed=False))
centr_clos = np.array(sm_gcn_nx.closeness())
# new measures
eccec = np.array(sm_gcn_nx.eccentricity())
pager = np.array(sm_gcn_nx.personalized_pagerank(directed=False))
const = np.array(sm_gcn_nx.constraint())
hubs = np.array(sm_gcn_nx.hub_score())
auths = np.array(sm_gcn_nx.authority_score())
coren = np.array(sm_gcn_nx.coreness())
diver = np.array(sm_gcn_nx.diversity())

# add node properties to new df

# cretae dataset
genes = P[genes_pred_idx]
sm_df = pd.DataFrame()

sm_df['clust'] = pd.Series(clust) # clustering
sm_df['deg'] = pd.Series(deg) # degree
sm_df['neigh_deg'] = pd.Series(neigh_deg) # average_neighbor_degree
sm_df['betw'] = pd.Series(centr_betw) # betweenness_centrality
sm_df['clos'] = pd.Series(centr_clos) # closeness_centrality
sm_df['eccec'] = pd.Series(eccec) # eccentricity
sm_df['pager'] = pd.Series(pager) # page rank
sm_df['const'] = pd.Series(const) # constraint
sm_df['hubs'] = pd.Series(hubs) # hub score
sm_df['auths'] = pd.Series(auths) # authority score
sm_df['coren'] = pd.Series(coren) # coreness
sm_df['diver'] = pd.Series(diver) # diversity

for i in range(dimensions):
  sm_df['emb_{0}'.format(i)] = pd.Series(embeddings[:,i])
for i in range(n_components):
  sm_df['emb_red_{0}'.format(i)] = pd.Series(embeddings_1d[:,i])
sm_df['emb_clust'] = pd.Series(yhat)

columns = list(sm_df.columns)
sm_df = scale_data(sm_df)
for i in terms_pred_idx:
  trm = F[i]
  sm_df[trm] = pd.Series(gene_by_go[:,i])
sm_df['Gene'] = pd.Series(genes)
sm_df.to_csv('data.csv', index=False)

f = time()
print('Time: {0:.2f}s'.format(f-s))
