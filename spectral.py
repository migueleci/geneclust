#!/usr/bin/env python3
# coding: utf-8

# Gene function prediction - Spectral clustering
# Miguel Romero - Oscar Ramirez, sep 30 2021

import os
import sys
import csv
import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from collections import Counter

# create path
def create_path(path):
  try:
    os.makedirs(path)
  except:
    pass


path = '...' # working path
file = open('{0}/genes.txt'.format(path), 'r')
genes = [x.strip() for x in file.readlines()]
ng = len(genes)
file.close()

data = pd.read_csv('{0}/gcn.csv'.format(path)) # or affg.csv
mat = np.zeros((ng, ng))

for u,v,s in tqdm(data.itertuples(index=False, name=None)):
	mat[u,v] = mat[v,u] = s
G_sparse = csr_matrix(mat)

for n_clusters in [10,20,30]:
  sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed_nearest_neighbors', n_jobs=64, random_state=0).fit(G_sparse)
  labels = sc.labels_

  file = open('{0}/spectral/{1}.csv'.format(path,n_clusters),'w') # if affg is used, change {1} for n{1}, or the files will be overwritten
  file.write('\n'.join([str(x) for x in labels]))
  file.close()
