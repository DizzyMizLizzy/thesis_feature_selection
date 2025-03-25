#! /usr/bin/env python3


import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import json
from pathlib import Path
import os
from tqdm import tqdm

@hydra.main(config_path="config", config_name="explain", version_base=None)   # read the config file (explain.yaml) under config.
def main(config: DictConfig) -> None:
    seed_everything(config.random_seed, workers=True)
    
  
    if config.is_lassonet:   # test lassonet model, lassonet model needs different initialization from the rest of methods.
        # read all models on the training path and keep the best results.
        model_path = Path(config.datasets.trained_fold)/ config.train_mode.mode/ config.models.trained_fold/ config.models.model_name
        config.train_mode.mask_fold = model_path

        test_loader = instantiate(config.train_mode.test_loader)

        NDCGs = []
        for i in tqdm(range(0,500,10)):
            checkpoint_path = model_path / f'lightning_logs/version_{i}' / 'checkpoints' 
            ckpt = list(checkpoint_path.glob('*.ckpt'))
            if ckpt:
                ckpt = str(ckpt[0])
            else:
                break

            if os.path.isfile(ckpt):
                model = instantiate(config.models.model).load_state_dicts(ckpt)
                trainer = pl.Trainer(strategy='ddp', detect_anomaly=True, gpus=config.gpus, deterministic=True)
                test_result = trainer.test(model, test_loader)
                ndcg10 = model.test_result[-1]['test_ndcg@10']
                NDCGs.append(ndcg10)

            else:
                break

        best_ndcg10 = max(NDCGs)
        NDCGs.append(best_ndcg10)
        with open(model_path / f'test_ndcg@10_{config.global_mask_ratio}.json', 'w')as f:
            json.dump(NDCGs, f)



    else:
        test_loader = instantiate(config.train_mode.test_loader)
        checkpoint = Path(config.checkpoint_path)
        ckpt = str(list(checkpoint.glob('*.ckpt'))[0])
        
        if os.path.isfile(checkpoint/config.test_result):
            print(f'Test results exist, skip.')
        else:
            print(f'Load model from {ckpt}.')
            model = instantiate(config.models.model).load_from_checkpoint(ckpt).double()
            model.eval()
            if config.global_mask.lower() == 'l2x':
                model.critic_only=True
                print(f'Use predictor only for L2X model.')
            
            # !!! strategy chose dp, automatically reduced outputs in epoch end function.
            trainer = pl.Trainer(strategy='ddp', detect_anomaly=True, gpus=config.gpus, deterministic=True)
            test_result = trainer.test(model, test_loader)
            save_test = checkpoint / f'{config.test_result}_{config.global_mask_ratio}.json'
            with open(save_test, 'w')as f:
                json.dump(model.test_result, f)
            print(f'Saved test results to {save_test}.')
            

        
if __name__ == '__main__':
    main()
