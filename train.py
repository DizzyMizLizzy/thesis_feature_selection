#! /usr/bin/env python3


import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import json
from pathlib import Path

@hydra.main(config_path="config", config_name="train", version_base=None)  # read the config file (train.yaml) under config.
def main(config: DictConfig) -> None:
    seed_everything(config.random_seed, workers=True)
    
    if config.global_mask.lower() == 'lassonet':
        config.train_mode.mask_fold = f'{config.datasets.trained_fold}/{config.train_mode.mode}/{config.mask_dir}'
    train_loader = instantiate(config.train_mode.train_loader)
    valid_loader = instantiate(config.train_mode.valid_loader)
    test_loader = instantiate(config.train_mode.test_loader)

    if config.is_lassonet:
        model = instantiate(config.models.model)
        model._run(train_loader, valid_loader, test_loader)
    else:
        model = instantiate(config.models.model).float()
        trainer = instantiate(config.trainer)
        trainer.fit(model, train_loader, valid_loader)
        test_result = trainer.test(model, test_loader, ckpt_path='best')
        save_test = Path(trainer.checkpoint_callback.best_model_path).parent / 'test_results.json'
        with open(save_test, 'w')as f:
            json.dump(model.test_result, f)
        
if __name__ == '__main__':
    main()