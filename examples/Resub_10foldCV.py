def get_cv(method,bool_trans,X,Y,n_splits=10,cv_type = "StratifiedKFold",verbose = True):  
  """Cross validation to get AUC.
  method: str, ['suliu', 'pepe', 'min-max','stepwise', 'logistic']
  X: design matrix
  Y: labels
  bool_trans: whether applied log transformation of X
  """
  if cv_type == "StratifiedKFold":
    cv = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state=42) # The folds are made by preserving the percentage of samples for each class.
  else: 
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
  
  model = AllMethod(method= method, bool_trans= bool_trans)
  #training_time = []
  AUC = []

  for folder, (train_index, val_index) in enumerate(cv.split(X, Y)): 
    X_train,X_val = X.iloc[train_index],X.iloc[val_index]
    y_train,y_val = Y.iloc[train_index],Y.iloc[val_index]
    # 
    X0_train, X1_train = helper(X_train, y_train); X0_val, X1_val = helper(X_val, y_val)

    model.fit(X0_train,X1_train)
    _,_, auc = model.predict(X0_val,X1_val)
    AUC.append(auc)
    if verbose:
        #print('estimated coef is %s' % model.coef_)
        print('fitted AUC is %s' % model.fiited_auc_)
        print("test auc in %s fold is %s" % (folder+1,auc) )
        print('____'*10)
  return AUC      


