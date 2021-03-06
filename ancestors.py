#!/usr/bin/env python3
# coding: utf-8

# Gene function prediction - Gene Ontology Term Enrichment and feature creation
# Miguel Romero - Oscar Ramirez, sep 30 2021

import os
import json
import numpy as np
import igraph as ig
import pandas as pd
from tqdm import tqdm
import dataframe_image as dfi
from matplotlib import pyplot as plt
from goatools.obo_parser import GODag
from goatools.semantic import deepest_common_ancestor, common_parent_go_ids
from goatools.godag.go_tasks import get_go2parents
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.semantic import semantic_similarity

import scipy.stats as stats                # for fisher test
import statsmodels.stats.multitest as smt  # for pvalue adjustment

pd.set_option("display.precision", 2)

godag = GODag("go-basic.obo")

ndf = pd.read_csv("gfunc.csv")
def array2list(str):
  str = str.replace("\'",'\"')
  return json.loads(str)
ndf['funcs'] = ndf.funcs.apply(lambda t: array2list(t))

ufunc = list()
for x in ndf['funcs'].tolist(): ufunc += x
ufunc = list(set(ufunc))

gene4func = dict([(x,0) for x in ufunc])
for x in ndf['funcs'].tolist():
  for f in x:
    gene4func[f] += 1

df_funcs = pd.DataFrame()
df_funcs['Function'] = gene4func.keys()
df_funcs['Description'] = [godag[x].name for x in df_funcs['Function']]
df_funcs['Genes'] = gene4func.values()
df_funcs = df_funcs.sort_values(['Genes'], ascending=False).reset_index(drop=True)


def GO_enrichment():
  '''
  :param com2genes: dictionary where keys are community ids and values are list of genes belonging to the corresponding community
  :param GO2genes: dictionary where keys are GO terms and values are list of genes with the corresponding GO annotation
  :param gene2GOs: dictionary where keys are genes and values are list of GO terms annotated to the corresponding gene
  :param n: total number of genes in the background set
  '''

  nldf = ldf[ldf['label'] >= 0]
  nbr_coms = len(nldf['label'].unique())

  ans = []
  for m in range(nbr_coms):
    module = nldf[nldf['label'] == m]
    nbr_genes_in_module = len(module)

    # GO terms present in the module
    module_GOs = list()
    mndf = ndf[ndf['gid'].isin(module.index)]
    for x in mndf['funcs'].tolist(): module_GOs += x
    module_GOs = set(module_GOs)

    for go in module_GOs:
      nbr_genes_in_GO = gene4func[go]

      a = np.sum([1 if go in x else 0 for x in mndf['funcs']])
      b = nbr_genes_in_GO - a
      c = nbr_genes_in_module - a
      d = len(ldf) - nbr_genes_in_GO - c

      _, pvalue = stats.fisher_exact([[a,b],[c,d]])

      # mod, go, pval
      ans.append([m,go,pvalue])

  goedf = pd.DataFrame(ans, columns=['module','GO_id','pvalue'])
  return goedf

def FDR(tmpdf):
  '''
  :param df: DataFrame from GO_enrichment function
  '''
  fdr = []
  coms = tmpdf.module.unique()
  for m in coms:
    try:
      pval_adj = smt.multipletests(pvals = tmpdf[tmpdf.module==m].pvalue, method = 'fdr_bh')[1]
      fdr += list(pval_adj)
    except:
      fdr += tmpdf[tmpdf.module==m].pvalue.tolist()

  tmpdf['fdr'] = fdr
  return tmpdf

def enriched_modules():
  goedf = GO_enrichment()
  goedf2 = FDR(goedf)
  goedf2 = goedf2[goedf2.fdr < 0.05]
  return goedf2
  # return goedf2.module.nunique()


'''
Create files to compare with xgb results

'''

# create file of pvalues
def create_files_pvalues(labels, df, folder):
  total = 0
  for func in df.GO_id.unique():
    arr = np.zeros(len(labels))
    for mod in df[df.GO_id==func].module:
      pvalue = df[(df.GO_id==func) & (df.module==mod)].pvalue
      idx = labels[labels.label==mod].index
      arr[idx] = 1 - pvalue
    print('{0} --> {1}'.format(func, np.count_nonzero(arr)))
    res = pd.DataFrame()
    res['label'] = arr
    res.to_csv('{0}/{1}.csv'.format(folder, func.replace(':','')), index=False)
    total += np.sum(arr)
  return total

def create_path(path):
  try:
    os.makedirs(path)
  except:
    pass

path = 'spectral'

n_clusters = [10,20,30,40,50,60,70,80,90,100]

files = ['n{0}.csv'.format(x) for x in n_clusters]
for nc, file in zip(n_clusters, files):
  ldf = pd.read_csv('{0}/{1}'.format(path,file), names=["label"])
  n = len(ldf['label'].unique())
  goedf2 = enriched_modules()

  outpath = '{0}/n{1}'.format(path, nc)
  create_path(outpath)
  total = create_files_pvalues(ldf, goedf2, outpath)

files = ['{0}.csv'.format(x) for x in n_clusters]
for nc, file in zip(n_clusters, files):
  ldf = pd.read_csv('{0}/{1}'.format(path,file), names=["label"])
  n = len(ldf['label'].unique())
  goedf2 = enriched_modules()

  outpath = '{0}/{1}'.format(path, nc)
  create_path(outpath)
  total = create_files_pvalues(ldf, goedf2, outpath)
