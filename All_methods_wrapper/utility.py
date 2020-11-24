#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
Created on Nov 23, 2020
@author: Chengning Zhang
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from numpy.random import multivariate_normal
from sklearn.svm import SVC
from scipy.stats import bernoulli

# Computing empirical AUC 
def indicator(s0,s1):
  """Indicator function
  s0: scalar
  s1: scalar
  return scalar
  """
  if s0 == s1: return 0.5
  elif s1 > s0: return 1
  else: return 0

def NonpAUC(s0,s1):
  """compute the nonparametruc AUC. 
  s1: array of composite scores for class '1'.
  s0: array of composite scores for class '0'
  return scalar auc
  """
  n1 = len(s1)
  n0 = len(s0)
  ans = 0
  for x1 in s1:
    for x0 in s0:
      ans += indicator(x0,x1)
  return ans/(n1*n0)
  
# anchor var's coef is 1. 
def anchor_est_coef(coef):
  """
  set the first var as anchor var, which has coef 1.
  coef: array(cannot be list)
  return coef_, array.
  """

  # if anchor coef <0
  coef_ = np.array(coef)
  return coef_/abs(coef_[0]) # abs guarantee it is increasing transformation
 
# SULIU
def suliu(X0, X1, bool = True):
  """
  X0: df or array, design matrix for class '0'
  X1: df or array, design matrix for class '1'
  """
  a = np.cov(X0, rowvar= False) +  np.cov(X1, rowvar= False)
  b = X1.mean().to_numpy() - X0.mean().to_numpy()
  est_coef = np.matmul(inv(a),b)
  if bool: 
    est_coef = anchor_est_coef(est_coef)
  # 
  Y0 = np.matmul(X0.to_numpy(), est_coef); Y1 = np.matmul(X1.to_numpy(), est_coef)
  auc = NonpAUC(Y0, Y1)
  if auc >=0.5:
    return est_coef, auc
  else:
    return -est_coef, 1-auc
    
## RF
def randomforst(X0,X1):
  n0 = X0.shape[0]
  n1 = X1.shape[0]

  X = pd.concat([X0,X1])
  y = [0] * n0
  y.extend([1]*n1); y = np.array(y)
  rf =RandomForestClassifier(max_depth=2, random_state=43).fit(X,y)
  ## 
  y_pred = rf.predict_proba(X)[:,1]
  auc = roc_auc_score(y, y_pred)
  #print(NonpAUC(y_pred[:n0], y_pred[n0:]))
  #feature_importances = rf.feature_importances_
  return rf, auc ## return model, for future prediction
  
# SVMr  
def svm_r(X0,X1):
  """svm with rbf kernel
  X0: df, design matrix for class '0'
  X1: df, design matrix for class '1'
  """
  n0 = X0.shape[0]
  n1 = X1.shape[0]

  X = pd.concat([X0,X1])
  y = [0] * n0
  y.extend([1]*n1); y = np.array(y)
  mod = SVC(kernel = 'rbf',random_state=42, probability= True).fit(X,y)
  ## 
  y_pred = mod.predict_proba(X)[:,1]
  auc = roc_auc_score(y, y_pred)

  return mod, auc ## cannot return estimates, so return mod for future prediction

# SVMl
def svm_l(X0,X1):
  """svm with linear kernel
  X0: df, design matrix for class '0'
  X1: df, design matrix for class '1'
  """
  n0 = X0.shape[0]
  n1 = X1.shape[0]

  X = pd.concat([X0,X1])
  y = [0] * n0
  y.extend([1]*n1); y = np.array(y)
  mod = SVC(kernel = 'linear',random_state=0, probability= True).fit(X,y)
  ## 
  y_pred = mod.predict_proba(X)[:,1]
  auc = roc_auc_score(y, y_pred)

  return mod, auc ## cannot return estimates, so return mod for future prediction

# LOGISTIC
def logistic(X0,X1, bool = True):
  n0 = X0.shape[0]
  n1 = X1.shape[0]

  X = pd.concat([X0,X1])
  y = [0] * n0
  y.extend([1]*n1); y = np.array(y)
  lr = LR(random_state=0).fit(X,y)
  ## 
  y_pred = lr.predict_proba(X)[:,1]
  auc = roc_auc_score(y, y_pred)
  est_coef = lr.coef_[0]
  if bool: 
    est_coef = anchor_est_coef(est_coef)

  if auc >=0.5:
    return est_coef, auc
  else:
    return -est_coef, 1-auc

# PEPE, basis of MIN_MAX, SW
def nonp_combine2_auc(l1,l2, X0, X1):
  """
  compute nonparametric AUC when X0 and X1 has two cols for given coef (l1,l2)
  l1: first coef
  l2: second coef
  X0: df, design matrix for class '0'
  X1: df, design matrix for class '1'
  """
  n0 = X0.shape[0]
  n1 = X1.shape[0]
  s0 = np.matmul(X0.to_numpy(), np.array([l1,l2]))
  s1 = np.matmul(X1.to_numpy(), np.array([l1,l2]))
  return NonpAUC(s0,s1)


def pepe(X0,X1, evalnum = 201, bool = True):
  """
  compute the coef that has max nonparametric AUC, X0 and X1 has two cols.
  X0: df, design matrix for class '0'
  X1: df, design matrix for class '1'
  """

  l = np.linspace(start=-1, stop=1, num=evalnum)
  #l2 = np.linspace(start=-1, stop=1, num=evalnum)
  auc_l1 = [nonp_combine2_auc(e,1,X0,X1) for e in l]  
  auc_l2 = [nonp_combine2_auc(1,e,X0,X1) for e in l] 
  if max(auc_l1) > max(auc_l2):
    ind = auc_l1.index(max(auc_l1))
    est_coef = np.array([l[ind],1])
    if bool: 
      est_coef = anchor_est_coef(est_coef)
    return est_coef, max(auc_l1)
  
  else:
    ind = auc_l2.index(max(auc_l2))
    est_coef = np.array([1,l[ind]])
    if bool: 
      est_coef = anchor_est_coef(est_coef)
    return est_coef, max(auc_l2)

# MIN_MAX
def liu(X0, X1, bool = True):
  """
  X0: df, design matrix for class '0'
  X1: df, design matrix for class '1'
  """
  # get min max row-wise
  max_min_X0 = np.concatenate( (  np.amax(X0.to_numpy(), axis=1).reshape(-1,1) , np.amin(X0.to_numpy(), axis=1).reshape(-1,1)  ), axis =1 ) 
  max_min_X1 = np.concatenate( (  np.amax(X1.to_numpy(), axis=1).reshape(-1,1) , np.amin(X1.to_numpy(), axis=1).reshape(-1,1)  ), axis =1 ) 
  max_min_X0 = pd.DataFrame(data = max_min_X0); max_min_X1 = pd.DataFrame(data = max_min_X1)
  return pepe(max_min_X0, max_min_X1, bool = bool)
  

# helper function for SW
def auc_check(X0, X1):
  """calculate AUC for every var
  X0: df, design matrix for class '0'
  X1: df, design matrix for class '1'
  """
  p = X0.shape[1]
  auc_list = []
  for i in list(X0.columns):
    auc_list.append(NonpAUC(X0.loc[:,i], X1.loc[:,i] ) )

  return auc_list
  
# SW
def stepwise(X0, X1, bool = True):
  n0 = X0.shape[0]
  n1 = X1.shape[0]
  varnum = X0.shape[1]
  combcoef = []
  ## step down
  auc_order = np.array(auc_check(X0,X1))
  sort_index = np.argsort(auc_order); sort_index= sort_index[::-1] ## auc_order[sort_index[0]] largest,  auc_order[sort_index[len()]] smallest
  combmarker0 = X0.iloc[:,[sort_index[0]]].copy() # pd
  combmarker1 = X1.iloc[:,[sort_index[0]]].copy() # pd
  nal_coef = [1]
  for i in range(1,varnum):
    #combmarker0 = pd.concat([combmarker0, X0.iloc[:, [sort_index[i]] ] ], axis= 1,ignore_index = True)
    #combmarker1 = pd.concat([combmarker1, X1.iloc[:, [sort_index[i]] ] ], axis= 1,ignore_index = True)
    combmarker0['new'] =  X0.iloc[:, [ sort_index[i] ] ].to_numpy() 
    combmarker1['new'] =  X1.iloc[:, [ sort_index[i] ] ].to_numpy()
    temp_inf , _ = pepe(combmarker0,combmarker1, bool= False)
    #print(temp_inf)
    combcoef.append(temp_inf)
    nal_coef = temp_inf[0]*np.array(nal_coef); nal_coef = list(nal_coef); nal_coef.append(temp_inf[1])
    combmarker0 = pd.DataFrame(data = np.matmul( combmarker0.to_numpy(),  temp_inf))
    combmarker1 = pd.DataFrame(data = np.matmul( combmarker1.to_numpy(),  temp_inf))
  
  est_coef = np.array([0.]*varnum) ## None has dtype problem, 0. makes float dtype. 
  for i in range(varnum):
    est_coef[sort_index[i]] = nal_coef[i]
  auc = NonpAUC( np.matmul(X0.to_numpy() ,est_coef ) , np.matmul(X1.to_numpy() ,est_coef ))
  
  if auc >=0.5:
    return est_coef, auc
  else:
    return -est_coef, 1-auc
    

# A Wrapper class for all methods above
class AllMethod:
  def __init__(self, method, bool_trans = True):
    """
    method: a string, specify which linear combination method to use. ['suliu', 'pepe', 'min-max','stepwise', 'rf', 'svml', 'svmr', 'logistic']
    bool_trans: whether to perform log transformation
    """
    self.method = method
    self.bool_trans = bool_trans
    
  def fit(self, X0, X1):
    """Train the model
    X0: df, design matrix for class '0'
    X1: df, design matrix for class '1'
    return: self, 
    obtain self.coef_ or self.mod, and self.fitted_auc 
    """
    if self.bool_trans:
      X0 = np.log(X0); X1 = np.log(X1)

    if self.method == 'suliu':
      self.coef_, self.fiited_auc_ = suliu(X0,X1,bool=False)
      
    elif self.method == 'logistic':
      self.coef_, self.fiited_auc_ = logistic(X0,X1,bool=False)

    elif self.method == 'min-max':
      self.coef_, self.fiited_auc_ = liu(X0,X1,bool=False)
    
    elif self.method == 'stepwise':
      self.coef_, self.fiited_auc_ = stepwise(X0,X1,bool=False)

    elif self.method == 'pepe':
      if X0.shape[1] != 2: 
        raise ValueError("Passed array is not of the right shape")
      self.coef_, self.fiited_auc_ = pepe(X0,X1,bool=False)

    elif self.method == 'svml':
      self.mod_, self.fiited_auc_ = svm_l(X0,X1)

    elif self.method == 'svmr':
      self.mod_, self.fiited_auc_ = svm_r(X0,X1)

    elif self.method == 'rf':
      self.mod_, self.fiited_auc_ = randomforst(X0,X1)
    return self

  def predict(self, X0, X1):
    """predict 
    X0: df, design matrix for class '0'
    X1: df, design matrix for class '1'
    return: y0, y1
    """

    if self.bool_trans:
      X0 = np.log(X0); X1 = np.log(X1)
    
    if self.method in ['rf', 'svml', 'svmr']: ## no self.coef_ , self.mod_
      y0 = self.mod_.predict_proba(X0)[:,1]
      y1 = self.mod_.predict_proba(X1)[:,1]
      auc = NonpAUC(y0, y1)
      return y0, y1, auc

    else: ## other methods, which return self.coef_
      if self.method == 'min-max':
        max_min_X0 = np.concatenate( (  np.amax(X0.to_numpy(), axis=1).reshape(-1,1) , np.amin(X0.to_numpy(), axis=1).reshape(-1,1)  ), axis =1 ) 
        max_min_X1 = np.concatenate( (  np.amax(X1.to_numpy(), axis=1).reshape(-1,1) , np.amin(X1.to_numpy(), axis=1).reshape(-1,1)  ), axis =1 ) 
        X0 = pd.DataFrame(data = max_min_X0); X1 = pd.DataFrame(data = max_min_X1)
      
      y0 = np.matmul(X0.to_numpy(), self.coef_ )
      y1 = np.matmul(X1.to_numpy(), self.coef_ )
      auc = NonpAUC(y0, y1)
      return y0, y1, auc

  
  def roc_plot(self, X0, X1):
    #if self.bool_trans:
    #  X0 = np.log(X0); X1 = np.log(X1) ## in self.predict, already did the transformation!

    n0 = X0.shape[0]; n1 = X1.shape[0]
    y0, y1, auc = self.predict(X0,X1); #print(y0); print(y1)
    y = [0] * n0
    y.extend([1]*n1); y = np.array(y); #print(y)
    y_pred = np.concatenate((y0,y1)); #print(y_pred)
    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1)
    plt.title('Receiver Operating Characteristic, Method: % s' % self.method)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    

# helper function to split X via Y into X0 and X1
def helper(X, Y):
  """Take X, Y, return X0 and X1
  X: df/array
  Y: df.series
  return X0, X1
  """
  #try:
  X0 = X.loc[Y == 0].copy()
  #except:
  #  X0 = X[Y == 0]
  #try:
  X1 = X.loc[Y == 1].copy()
  #except:
    #X1 = X[Y == 1]
  return X0,X1
  

