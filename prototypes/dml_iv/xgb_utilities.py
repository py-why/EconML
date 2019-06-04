from sklearn.model_selection import train_test_split
import copy

class XGBWrapper:

    def __init__(self, XGBoost, early_stopping_rounds, eval_metric, val_frac=0.1, binary=False):
        self.XGBoost = XGBoost
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.binary = binary
        self.val_frac = val_frac
        return

    def fit(self, X, y, sample_weight=None):
        if self.binary:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_frac,
                                                              random_state=123, stratify=y, shuffle=True)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_frac,
                                                              random_state=123, shuffle=True)
        self.XGBoost.fit(X_train, y_train, sample_weight=sample_weight, 
                         eval_set=[[X_val,y_val]], eval_metric=self.eval_metric,
                         early_stopping_rounds=self.early_stopping_rounds, verbose=False)
        return self

    def predict_proba(self, X):
        return self.XGBoost.predict_proba(X)

    def predict(self, X):
        return self.XGBoost.predict(X)

    def __getattr__(self, name):
        if name == 'get_params':
            raise AttributeError("not sklearn")
        return getattr(self.XGBoost, name)

    def __deepcopy__(self, memo):
        return XGBWrapper(copy.deepcopy(self.XGBoost, memo), 
                          self.early_stopping_rounds, self.eval_metric,
                          val_frac=self.val_frac, binary=self.binary)