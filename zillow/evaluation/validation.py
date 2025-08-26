from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import pandas as pd
import numpy as np
import datetime


class Submission:
    def __init__(self, pred, id) -> None:
        self.submission = None
        self._handle_pred(pred, id)


    def _handle_pred(self, pred, id):
        if pred.shape[1] == 1:
            self._compose_broadcast_submission(pred, id)

    def _compose_broadcast_submission(self, ypred, id):
        columns = ['201610', '201611', '201612', '201710', '201711', '201712']
        self.submission = pd.DataFrame(np.tile(ypred, (len(columns), 1)).T, columns=columns)
        self.submission['ParcelId'] = id

    def save_submission(self, filename="", path="submissions/", dtypes='default'):
        if not filename:
            filename = get_current_time() + ".csv"
        path += filename

        if dtypes:
            if dtypes == 'default':
                dtypes = {
                    'ParcelId': np.int32, 
                    '201610': np.float16, 
                    '201611': np.float16, 
                    '201612': np.float16, 
                    '201710': np.float16, 
                    '201711': np.float16, 
                    '201712': np.float16
                }

            self.submission = self.submission.astype(dtypes)

        self.submission.to_csv(path, index=False)

def get_current_time():
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d_%H-%M-%S")

def compute_metrics_default(ypred, ytrue):
    mae = mean_absolute_error(ytrue, ypred)
    r2 = r2_score(ytrue, ypred)
    return {"mae": mae, "r2": r2}

class Validation:
    def __init__(self, train, val, model, target='logerror',
                 metric='neg_mean_absolute_error', cv=5):
        self.train = train
        self.val = val

        # Order matters for tm split!
        self.data = pd.concat([train, val])
        self.model = model
        self.target = target
        self.metric = metric
        self.cv = cv

    def run_split(self, custom_metric=compute_metrics_default):
        self.model.fit(self.train.drop(columns=self.target), self.train[self.target])
        ypred = self.model.predict(self.val.drop(columns=self.target))
        return custom_metric(ypred, self.val[self.target])

    def run_cross_validation(self):
        return cross_val_score(self.model, self.data.drop(columns=self.target), 
                               self.data[self.target], cv=self.cv, scoring=self.metric)

    def run_tm_split(self):
        tscv = TimeSeriesSplit(n_splits=self.cv)
        res = np.zeros(self.cv)
  
        for i, (train_index, val_index) in enumerate(tscv.split(self.data)):
            self.model.fit(self.data.iloc[train_index].drop(columns=self.target), 
                           self.data.iloc[train_index][self.target])
            
            ypred = self.model.predict(self.data.iloc[val_index].drop(columns=self.target))
            res[i] = compute_metrics_default(ypred, self.data.iloc[val_index][self.target])
        
        return res