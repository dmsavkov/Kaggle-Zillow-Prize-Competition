import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

def init_adv_cv(model_name, seed, verbose=0, **kwargs):
    verbose = 0
    seed = seed

    match model_name:
        case 'logistic':
            model = LogisticRegression(random_state=seed, verbose=verbose, **kwargs) 
        case 'lgbm': 
            model = LGBMClassifier(random_state=seed, verbose=[-1, 1][verbose > 0], **kwargs)
        case _:
            raise ValueError(f"Unknown model name: {model_name}. Supported: 'logistic', 'lgbm'.")

    return AdversarialCV(
        model=model,
        rseed=seed,
        verbose=verbose
    )

class AdversarialCV:
    def __init__(self, model, rseed=42, verbose=1):
        self.model = model
        self.rseed = rseed
        self.verbose = verbose

    def make_cv(self, df1, df2, cv=5, shuffle=True, fold_method=None):
        model = clone(self.model)
        res = np.zeros(shape=cv)
        common_cols = list(set(df1.columns) & set(df2.columns))
        # TODO fix and adapt importances

        df1 = df1[common_cols].copy()
        df2 = df2[common_cols].copy()

        df1["test"] = 0
        df2["test"] = 1

        df = pd.concat([df1, df2], axis=0)
        X, y = df.drop("test", axis=1), df["test"]
        
        if isinstance(fold_method, str):
            if fold_method == 'skf':
                fold_method = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=self.rseed)
        
        else: # including 'kf'
            fold_method = KFold(n_splits=cv, shuffle=shuffle, random_state=self.rseed)
       
        importances = np.zeros(len(common_cols))
        for i, (train_ix, test_ix) in enumerate(fold_method.split(X, y)):
            model.fit(X.iloc[train_ix], y.iloc[train_ix])
            
            if hasattr(model, 'predict_proba'):
                yhat = model.predict_proba(X.iloc[test_ix])[:, 1]
            else:
                yhat = model.predict(X.iloc[test_ix])
                
            res[i] = roc_auc_score(y.iloc[test_ix], yhat)

            #importances += model.feature_importances_
        
        if self.verbose > 0:
            #importances = np.round(importances / cv, 2)
            #print("Importances for the iteration: ")
            #print(importances, df.columns[importances.argsort()[::-1][:20]])
            pass

        return res