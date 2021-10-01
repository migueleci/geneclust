#!/usr/bin/env python
# coding: utf-8

# Gene function prediction - Plot function
# Miguel Romero - Oscar Ramirez, sep 30 2021

import os
import sys
import numpy as np
import pandas as pd

# Ploting
import seaborn as sns
from matplotlib import rc
from matplotlib import pyplot as plt

rc('font', family='serif', size=18)
rc('text', usetex=False)

# Default colors
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#7f7f7f', '#bcbd22', '#17becf']



# plot auc roc curve
def plot_roc(term, folder, fpr, tpr, auc):
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  plt.plot(fpr, tpr, lw=2, label='AUC = {0:.2f}'.format(auc))
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('1 - Specificity')
  plt.ylabel('Sensitivity')
  plt.legend(loc='lower right')
  plt.savefig('{0}/{1}_roc.pdf'.format(folder, term), format='pdf', dpi=600)
  plt.close()

# plot multiple auc roc curves in one figure
def plot_mroc(term, folder, fprs, tprs, aucs, lbs):
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  for fpr, tpr, auc, lb in zip(fprs, tprs, aucs, lbs):
    plt.plot(fpr, tpr, lw=2, label='{0} AUC = {1:.2f}'.format(lb, auc))
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('1 - Specificity')
  plt.ylabel('Sensitivity')
  plt.legend(loc='lower right')
  plt.savefig('{0}/{1}_roc.pdf'.format(folder, term), format='pdf', dpi=600)
  plt.close()

# plot average precision curve
def plot_ap(term, folder, recall, precision, ap):
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  plt.plot(recall, precision, lw=2, label='AP = {0:.2f}'.format(ap))
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc='lower right')
  plt.savefig('{0}/{1}_pr.pdf'.format(folder, term), format='pdf', dpi=600)
  plt.close()

# plot multiple average precision curves in one figure
def plot_map(term, folder, recalls, precisions, aps, lbs):
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  for recall, precision, ap, lb in zip(recalls, precisions, aps, lbs):
    plt.plot(recall, precision, lw=2, label='{0} AP = {1:.2f}'.format(lb, ap))
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc='lower right')
  plt.savefig('{0}/{1}_pr.pdf'.format(folder, term), format='pdf', dpi=600)
  plt.close()

# plot line with std dev
def plot_mts_hist(x,y,e,folder,name):
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  plt.plot(x, y, '--')
  lowerb = [yi-ei for yi,ei in zip(y,e)]
  upperb = [yi+ei for yi,ei in zip(y,e)]
  plt.fill_between(x, lowerb, upperb, alpha=.3)
  plt.ylabel(name)
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.savefig('{0}/{1}_hist.pdf'.format(folder,name), format='pdf', dpi=600)
  plt.close()

# plot pie
def plot_feat_imp(l,x,folder,name):
  fig, ax = plt.subplots(figsize=(8,5))
  plt.pie(x, autopct='%1.1f%%', textprops=dict(size=14, color='w', weight='bold'))
  plt.legend(l, loc='best')
  plt.axis('equal')
  plt.tight_layout()
  plt.savefig('{0}/{1}_fimp.pdf'.format(folder,name), format='pdf', dpi=600)
  plt.close()

# plot confusion matrix
def plot_conf_matrix(cm, folder, name="", axis=[0,1], normalize=True):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  df_cm = pd.DataFrame(cm, index=axis, columns=axis)
  fig, ax = plt.subplots(figsize=(5,5))
  if normalize:
    sns.heatmap(df_cm, annot=True, cbar=False, linewidths=.5, center=0, cmap=plt.cm.Blues)
    figname = '{0}/{1}_norm.pdf'.format(folder,name)
  else:
    sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, linewidths=.5, center=0, cmap=plt.cm.Blues)
    figname = '{0}/{1}.pdf'.format(folder,name)
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.tight_layout()
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()

# plot auc roc vs height in hierarchy
def plot_auc_height(data,folder,name):
  x, y = [x for x,y in data], [y for x,y in data]
  fig, ax = plt.subplots(figsize=(5,5))
  plt.plot(x,y,'.')
  plt.xlabel('Height')
  plt.ylabel('AUC ROC')
  plt.tight_layout()
  plt.savefig('{0}/{1}_auc_height.pdf'.format(folder,name), format='pdf', dpi=600)
  plt.close()

# plot multiple line plots
def line_plot(data,model_name,xlabel,ylabel,xticks,ylim,path):
  fig, ax = plt.subplots(figsize=(8,8))
  x = np.arange(len(xticks))
  for idx, (_data, _model_name) in enumerate(zip(data,model_name)):
    plt.plot(x, _data, 'b-', color=COLORS[idx], label=_model_name)
  plt.ylim(ylim)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.xticks(x, xticks, rotation=90)
  if len(model_name) > 1: plt.legend(loc='best')
  plt.tight_layout()
  name = "".join(filter(str.isalnum, ylabel)).lower()
  plt.savefig('{0}/{1}.pdf'.format(path, name), format='pdf', dpi=600)
  # plt.show()
  plt.close()
