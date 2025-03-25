
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import RFE
import lightgbm as lgb
from pathlib import Path
import pickle


""" Feature selection using RFE, for lambdamart model. """
class FeatureSelect:
    def __init__(self, hparams: dict) -> None:
        self.feature_num = hparams['feature_num']
        self.fold_path = hparams['fold_path']
        self.model_path = hparams['model_path']
        self.estimater = self.load_model(hparams['tree_params'])
        self.rfe = RFE(self.estimater, n_features_to_select=self.feature_num)

    def load_model(self, params: dict):
        model = lgb.LGBMRanker(**params)
        return model

    def load_data(self, fold_path: Path, data_type: str):
        fname = str(fold_path / f'{data_type}.txt')    
        fgroup = fold_path / f'{data_type}_query_group.pkl'
        X, Y, _ = load_svmlight_file(fname, query_id=True)     # only accept str, not Path.
        X = X.toarray()
        with open(fgroup, 'rb')as f:
            group = pickle.load(f)

        return X, Y, group

    def feature_select(self, indices: bool=False, retrain: bool=True):
        """ Turn indices to True to get values, otherwise a boolean mask"""
        X_train, Y_train, group_train = self.load_data(self.fold_path, 'train')
        X_valid, Y_valid, group_valid = self.load_data(self.fold_path, 'valid') 
        extra = {'group': group_train, 'eval_set':[(X_valid, Y_valid)], 'eval_group':[group_valid]}
        self.rfe = self.rfe.fit(X_train, Y_train, **extra)
        if retrain:
            self.retrain(X_train, Y_train, extra)
        return self.rfe.get_support(indices=indices)
    
    def retrain(self, X, Y, extra_params):
        X_reduced = self.rfe.transform(X)
        self.estimater = self.estimater.fit(X_reduced, Y, **extra_params)
        self.estimater.booster_.save_model(self.model_path, num_iteration=self.estimater.booster_.best_iteration)


       