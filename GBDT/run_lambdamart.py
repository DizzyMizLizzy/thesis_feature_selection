import sys
sys.path.append('..')
import data_utils, metrics
from sklearn.datasets import load_svmlight_file
import argparse
from pathlib import Path
from lambdamart import LambdaMART
from feature_selection import FeatureSelect
import pandas as pd
import json
import time


dataset_dimension = {'Web30k': 136, 
                     'Yahoo': 699,
                     'MQ2008': 46

}

lambdamart_params = {'boosting_type': 'gbdt',
                        'objective': 'rank_xendcg',
                        #'objective': 'lambdarank',
                        'metric': 'ndcg',
                        #'max_bin': 64,
                        'num_trees': 300,  # 300,
                        'num_leaves': 500,   # == max_depth
                        'learning_rate': 0.1,
                        'num_iterations': 500,
                        #'early_stopping_rounds': 500,
                        'min_data_in_leaf': 1,
                        'min_sum_hessian_in_leaf': 100,
                        'num_thread': 16,
                        'verbose': -1,
                        'verbose_eval': True,
                        #'eval_at': 10,
                        'ndcg_eval_at':[1, 3, 5, 10],
                        'device': 'cpu',
                        'random_state': 123}


    
def main():
    start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--DIRECTORY', type=str, help='Your project directory.')
    ap.add_argument('--Dataset', type=str, help='MQ2008, MSLR30, Yahoo.')
    ap.add_argument('--Fold', type=str, default='Fold1', help='which fold?, within data_dir.')
    ap.add_argument('--task', type=str, default='train', help='train or test or rfe')
    ap.add_argument('--batch', type=int, default=500, help='Batch size for h5.') 
    ap.add_argument('--objective', type=str,  help='which objective? ')
    ap.add_argument('--feature_ratio', type=float, default=0.1, help='feature ratio.')
    ap.add_argument('--random_seed', type=int, default=123, help='random_seed')


    args = ap.parse_args()

    fold_dir = Path(args.DIRECTORY) / 'Datasets' / args.Dataset / args.Fold
    if args.task == 'train':
        """ Train Lambdamart model with fixed parameters. """
        args.save_model = Path(args.DIRECTORY) / 'Trained' / args.Dataset / args.Fold / 'lambdamart'/ f'{args.objective}_{args.random_seed}.txt'
        train_data = data_utils.load_h5_lgb(fold_dir, 'train', args.batch)
        valid_data = data_utils.load_h5_lgb(fold_dir, 'valid', args.batch, reference=train_data)
       
        args.train_params = lambdamart_params.copy()
        args.train_params.update({'objective': args.objective, 'random_state': args.random_seed})
        print(args.train_params)
        args.train_data = train_data
        args.valid_data = valid_data
        Lambda = LambdaMART(args)

        # test
        f_test = fold_dir / 'test.txt'
        X_test, Y_test, qids = load_svmlight_file(str(f_test), query_id=True)
        Y_pred = Lambda._predict(X_test.toarray())
        predict = [(label, score) for label, score in zip(Y_test, Y_pred)]
        Test_frame = pd.DataFrame({'qid': qids, 'predict': predict})
        NDCG = metrics.ndcg_score(Test_frame, k=args.train_params['ndcg_eval_at'])
        print(f'NDCG@1: {round(NDCG[-1][0], 3)}, NDCG@3: {round(NDCG[-1][1], 3)}, NDCG@5: {round(NDCG[-1][2],3)}, NDCG@10:{round(NDCG[-1][3],3)}.')
        test_json_file = args.save_model.parent / f'{args.objective}_{args.random_seed}_test_ndcg.json'
        with open(test_json_file, 'w')as f:
            json.dump(NDCG, f)

    elif args.task == 'rfe':
        """ Feature selection with RFE. """
        args.save_model = Path(args.DIRECTORY) / 'Trained' / args.Dataset / args.Fold / f'{args.model_name}_feature_{args.feature_ratio}.txt'
        save_feature = args.save_model.parent / f'{args.model_name}_feature_{args.feature_ratio}.json'
        hparams = {'feature_num': int(dataset_dimension[args.Dataset]*args.feature_ratio),
                    'fold_path': fold_dir, 
                    'model_path': args.save_model,
                    'tree_params': lambdamart_params}
        FS = FeatureSelect(hparams)
        top_feature = FS.feature_select(True, True)   # save selected features to indices, also retrain the model with selected features.
        with open(save_feature, 'w')as f:
            json.dump(top_feature.tolist(), f)
        print(f'Saved the top {args.feature_ratio} to {save_feature}')
    
    end = time.time()
    hours = (end - start)/3600
    print(f'Took {hours} hours to finish.')


if __name__ == '__main__':
    main()
