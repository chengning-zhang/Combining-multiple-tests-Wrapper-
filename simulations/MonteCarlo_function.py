#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
Created on Nov 23, 2020
@author: Chengning Zhang
"""

## simulation for Scenario A: generate X0 and X1. 
def MonteCarlo_1(T, n0, n1, u0, u1, sigma0, sigma1, log_bool = False):
  """simulation for first scenario: multivarite normal with equal variance
  T: number of simulation
  n0: sample size of class 0
  n1: sample size of class 1
  """
  AUC = {'suliu':[], 'logistic':[], 'stepwise':[],'min-max':[], 'rf':[], 'svml':[], 'svmr':[]}  ## same num as simulation time
  methods = ['suliu', 'logistic', 'stepwise','min-max', 'rf', 'svml', 'svmr']
  for i in range(T):
    ### one monto carlo simulation of size n0 + n1
    #i = 10
    np.random.seed(seed= 100*i+ 4*i)
    X0 = multivariate_normal(u0, sigma0, size = n0)
    X1 = multivariate_normal(u1, sigma1, size = n1)
    if log_bool:
      X0 = np.exp(X0)
      X1 = np.exp(X1)
    #
    X = np.concatenate([X0,X1])
    y = [0] * n0
    y.extend([1]*n1); y = np.array(y) ## X,y is one simulation
    X = pd.DataFrame(data = X); y = pd.Series(y)
    ## within that particular MC simulation, do 10 folds CV
    cv = StratifiedKFold(n_splits= 10, shuffle=True, random_state=42)
    AUC_folds = {'suliu':[], 'logistic':[], 'stepwise':[],'min-max':[], 'rf':[], 'svml':[], 'svmr':[]}  # same number as folders
    #
    for folder, (train_index, val_index) in enumerate(cv.split(X, y)): 
        X_train,X_val = X.iloc[train_index],X.iloc[val_index]
        y_train,y_val = y.iloc[train_index],y.iloc[val_index]
        # 
        X0_train, X1_train = helper(X_train, y_train); X0_val, X1_val = helper(X_val, y_val)
        for method in methods:
          model = AllMethod(method= method, bool_trans= False).fit(X0_train,X1_train)
          _,_, auc = model.predict(X0_val,X1_val)
          AUC_folds[method].append(auc)
    #print(AUC_folds)
    for key, val in AUC_folds.items():
      AUC[key].append( np.mean(np.array(val) ))

  print({key: (np.mean(np.array(val)) ,np.std(np.array(val))) for key,val in AUC.items()})
  return AUC
  
 
## Simulation scenario B: generate X first, then generate bernulli Y via logit(P(Y=1|X)) = ...
def MonteCarlo_3(T, n, u, sigma):
  """simulation for last scenario: generate X first from normal, then generate y via logit(Y|X) = 10* ((sinpi*x1) + ... )
  T: number of simulation
  n: sample size
  """
  AUC = {'suliu':[], 'logistic':[], 'stepwise':[],'min-max':[], 'rf':[], 'svml':[], 'svmr':[]}  ## same num as simulation time
  methods = ['suliu', 'logistic', 'stepwise','min-max', 'rf', 'svml', 'svmr']
  for i in range(T):
    ### one monto carlo simulation of size n0 + n1
    np.random.seed(seed= 100*i+ 4*i)
    X = multivariate_normal(u, sigma, size = n); #X = np.exp(X)
    X_trans = [ele[0] - ele[1] - ele[2]+ (ele[0] - ele[1])**2 - ele[3]**4 for ele in X] ## x1 - x2 - x3 + (x1-x2)^2 - x4^4
    p = list(map(lambda x: 1 / (1 + np.exp(-x)),  X_trans))
    y = bernoulli.rvs(p, size= n)
    X = pd.DataFrame(data = X); y = pd.Series(y)
    ## within that particular MC simulation, do 10 folds CV
    cv = StratifiedKFold(n_splits= 10, shuffle=True, random_state=42)
    AUC_folds = {'suliu':[], 'logistic':[], 'stepwise':[],'min-max':[], 'rf':[], 'svml':[], 'svmr':[]}  # same number as folders
    #
    for folder, (train_index, val_index) in enumerate(cv.split(X, y)): 
        X_train,X_val = X.iloc[train_index],X.iloc[val_index]
        y_train,y_val = y.iloc[train_index],y.iloc[val_index]
        # 
        X0_train, X1_train = helper(X_train, y_train); X0_val, X1_val = helper(X_val, y_val)
        for method in methods:
          model = AllMethod(method= method, bool_trans= False).fit(X0_train,X1_train)
          _,_, auc = model.predict(X0_val,X1_val)
          AUC_folds[method].append(auc)
    #print(AUC_folds)
    for key, val in AUC_folds.items():
      AUC[key].append( np.mean(np.array(val) ))

  print({key: (np.mean(np.array(val)) ,np.std(np.array(val))) for key,val in AUC.items()})
  return AUC
