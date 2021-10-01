#!/usr/bin/env python3
# coding: utf-8

# Gene function prediction - Supervised learning for gene function prediction
# Miguel Romero - Oscar Ramirez, sep 30 2021

import os
import sys
import datetime
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from time import time

# Own Libraries
from plots import *

# Metrics
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# Cross-validation and scaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

# Over-sampling and classifier Libraries
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore',category=DeprecationWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)

##################################################
# 1. Gene function prediction (biological process)
##################################################

# create path
def create_path(path):
  try:
    os.makedirs(path)
  except:
    pass


# SMOTE Technique (Over-Sampling)
def training_smote(X, y, term, n_splits, seed):
  # List to append the score and then find the average
  auc, ap, loss = list(), list(), list()
  fpr, tpr = list(),list()
  pre, rcl = list(),list()
  y_pred = np.zeros(len(y))
  y_pred_prob = np.zeros(len(y))

  clf = xgb.XGBClassifier(booster='gbtree', n_jobs=n_jobs_clf, random_state=seed,
                          eval_metric="aucpr") #, use_label_encoder=False)
  rand_xgb = RandomizedSearchCV(clf, param_grid, scoring="recall",
                                n_jobs=n_jobs_cv, n_iter=n_iter, random_state=seed)

  # Implementing SMOTE Technique, Cross Validating the right way
  sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
  for train_index, test_index in sss.split(X, y):
    Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
    ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

    Xtrain, Xtest = Xtrain.values, Xtest.values # Turn into an array
    ytrain, ytest = ytrain.values, ytest.values

    if np.sum(ytest) == 0 or np.sum(ytest) == len(ytest): continue
    if np.sum(ytrain) == 0 or np.sum(ytrain) == len(ytrain): continue

    try:
      pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_xgb)
      model = pipeline.fit(Xtrain, ytrain)
    except:
      pipeline = imbalanced_make_pipeline(RandomOverSampler(sampling_strategy='minority'), rand_xgb)
      model = pipeline.fit(Xtrain, ytrain)

    best_est = rand_xgb.best_estimator_
    prediction = best_est.predict(Xtest)
    pred_proba = best_est.predict_proba(Xtest)[:,1]
    y_pred[test_index] = prediction # save results per fold
    y_pred_prob[test_index] = pred_proba

    ap.append(average_precision_score(ytest, pred_proba))
    auc.append(roc_auc_score(ytest, pred_proba))
    loss.append(log_loss(ytest, pred_proba))

  return y_pred, y_pred_prob


def write_pred(y, figs_path, filename):
  file = open('{0}/{1}.csv'.format(figs_path,filename), 'w')
  file.write('\n'.join(['{0}'.format(i) for i in y]))
  file.close()


def compute_performance(y_orig, y_pred, y_pred_prob, figs_path, filename, plot=False):

  # ROC AUC
  roc = roc_auc_score(y_orig, y_pred_prob)
  fpr, tpr, _ = roc_curve(y_orig, y_pred_prob)

  # Average precision
  ap = average_precision_score(y_orig, y_pred_prob)
  prec, recall, thresh = precision_recall_curve(y_orig, y_pred_prob)

  # compute best threshold, according to PR curve
  fscore = (2*prec*recall)/(prec+recall)
  ix = np.argmax(fscore)
  y_pred_new = y_pred_prob.copy()
  y_pred_new[y_pred_new >= thresh[ix]] = 1
  y_pred_new[y_pred_new < thresh[ix]] = 0
  cm = confusion_matrix(y_orig, y_pred_new)
  cm = confusion_matrix(y_orig, y_pred_new, normalize='true')

  # F1 score
  f1s = f1_score(y_orig, y_pred_new)

  prec, recall, fscore, support = precision_recall_fscore_support(y_orig, y_pred_new)#, beta=beta)

  if plot:
    plot_roc(filename, figs_path, fpr, tpr, roc)
    print('aucROC: {0:.3f}'.format(roc))

    plot_ap(filename, figs_path, recall, prec, ap)
    print('avgPrec (aucPR): {0:.3f}'.format(ap))

    plot_conf_matrix(cm, figs_path, name=filename)
    plot_conf_matrix(cm, figs_path, name=filename, normalize=False)

    print('F1score: {0:.3f}'.format(f1s))
    print('Precision: {0:.3f}'.format(prec[1]))
    print('Recall: {0:.3f}'.format(recall[1]))

  return roc, ap, f1s, thresh[ix], prec[1], recall[1], cm[0,0], cm[1,1], y_pred_new


def readListFile(filename):
  file = open(filename, 'r')
  tmp = [x.strip() for x in file.readlines()]
  file.close()
  return tmp


def write_csv():
  df_resume = pd.DataFrame()
  df_resume['Term'] = pd.Series(term_list)
  df_resume['Genes'] = pd.Series(Genes)

  for key, val in zip(['roc','ap','f1s','thrh','prec','rec','time','tpr','tnr'],
                      [Aucroc,Ap,F1s,Thresh,Prec,Recall,Time,TPr,TNr]):
    df_resume[key] = pd.Series(val)

  df_resume.to_csv('{0}/resume.csv'.format(OUTPUT_PATH), index=False)


##################################
##################################
start_time = time()
dt = datetime.datetime.today()

PATH = "..." # working path

seed = 92021  # seed for random state
n_splits = 5 # number of folds
n_iter = 5   # n_iter for xgboost
cpu = multiprocessing.cpu_count()
n_jobs_cv = cpu // 2
n_jobs_clf = 2 # cpu // n_jobs_cv

param_grid = {
        'max_depth': [3, 6, 10],
        'min_child_weight': [0.5, 3.0, 5.0, 8.0],
        'eta': [0.01, 0.05, 0.2, 0.4],
        'subsample': [0.5, 0.7, 0.9, 1.0]}


###
# prediction
###

N = 10
M = N // 2

data = pd.read_csv('{0}/data.csv'.format(PATH), dtype='float')
data = data.drop(['Gene', 'diver'], axis=1)

Aucroc = list() #  auc roc score
Ap = list() # average precision score (auc pr)
F1s = list() # f1 score
Thresh = list() # Optimum threshold of PR cruve
Prec = list() # precision score
Recall = list() # recall score
Time = list() # execution time
TPr = list()
TNr = list()

Genes = list() # execution time

OUTPUT_PATH = "{0}/featsel/{1}-{2}-{3}".format(PATH, dt.year, dt.month, dt.day)
create_path(OUTPUT_PATH)

nclusters = [str(x) for x in range(10,101,10)] + ['n{0}'.format(x) for x in range(10,101,10)]

c = 0
file = open('featsel/top_feat.txt','r')
for l in file.readlines():
  c = max(c,len(l.split(',')))
nc_cols = ['c{0}'.format(x) for x in range(1,c)]
go_cols = [x for x in data.columns if 'GO:' in x]
terms_df = pd.read_csv('featsel/top_feat.txt', names=['term']+nc_cols, dtype=object)
term_list = terms_df.term.tolist()

for ridx, term in enumerate(term_list):
  figs_path = "{0}/{1}".format(OUTPUT_PATH, term.replace(':',''))
  # create_path(figs_path)

  print('### {0} of {1}, {2}'.format(ridx+1, len(term_list), term))
  Genes.append(data[term].sum())

  X = data[go_cols].copy()
  cols = terms_df[terms_df.term==term][nc_cols].values.tolist()[0]
  cols = [x for x in cols if x is not np.NaN]
  for c in cols:
    if c in nclusters:
      cldf = pd.read_csv('{0}/spectral/{1}/{2}.csv'.format(PATH, c, term.replace(':','')))
      X[c] = cldf.label

  X = X.drop([term], axis=1)
  y = data[term]

  tmp_auc = list()
  tmp_ap = list()
  tmp_f1s = list()
  tmp_thr = list()
  tmp_pre = list()
  tmp_rec = list()
  tmp_tim = list()
  tmp_tpr = list()
  tmp_tnr = list()
  tmp_pred = np.zeros(len(y))
  for i in tqdm(range(N)):
    s = time()
    pred, pred_prob = training_smote(X, y, term, n_splits, seed)

    # Evaluation of term
    # compute performance metrics for trial
    roc, ap, fsc, thr, pre, rec, tnr, tpr, pred_new = compute_performance(
      y, pred, pred_prob, figs_path, term.replace(':',''))
    tmp_pred += pred_new
    f = time()

    tmp_auc.append(roc)
    tmp_ap.append(ap)
    tmp_f1s.append(fsc)
    tmp_thr.append(thr)
    tmp_pre.append(pre)
    tmp_rec.append(rec)
    tmp_tpr.append(tnr)
    tmp_tnr.append(tpr)
    tmp_tim.append(f-s)

  Aucroc.append(np.mean(tmp_auc))
  Ap.append(np.mean(tmp_ap))
  F1s.append(np.mean(tmp_f1s))
  Thresh.append(np.mean(tmp_thr))
  Prec.append(np.mean(tmp_pre))
  Recall.append(np.mean(tmp_rec))
  TNr.append(np.mean(tmp_tpr))
  TPr.append(np.mean(tmp_tnr))
  Time.append(np.mean(tmp_tim))
  tmp_pred[tmp_pred<M] = 0
  tmp_pred[tmp_pred>=M] = 1

  write_csv()

print("--- {0:.2f} seconds ---".format(time() - start_time))
