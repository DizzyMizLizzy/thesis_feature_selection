from typing import Any, Dict
import lightgbm as lgb
import argparse


class LambdaMART:
    def __init__(self, args: argparse.Namespace) -> None: 
        self.evals_result = {}
        if args.task == 'train':
            self._set_params(args.train_params)
            self._train(args.train_data, args.valid_data, args.save_model)
            self.ranker = self._load_model(args.save_model)
        
        elif args.task == 'test':
            self.ranker = self._load_model(args.save_model)
        else:
            raise ValueError('task: train, test.')

    def _set_params(self, params: Dict[str, Any]=None):
        self.params = params.copy()

    def _train(self, train_data, valid_data, save_model_path: str):
        ranker = lgb.train(self.params, train_data, valid_sets=[valid_data], evals_result=self.evals_result)
        ranker.save_model(save_model_path, num_iteration=ranker.best_iteration)
    
    def _load_model(self, save_model_path: str):
        ranker = lgb.Booster(model_file= save_model_path)
        return ranker

    def _predict(self, data):
        ypred = self.ranker.predict(data, num_iteration=self.ranker.best_iteration)
        return ypred
    
    









