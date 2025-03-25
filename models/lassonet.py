from typing import Any, Dict
import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import StackLinear
from itertools import islice
import pytorch_lightning as pl
import numpy as np
import json
from pathlib import Path
import os
from base_ranker import BaseRanker
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

epsilon = torch.finfo(torch.float32).eps
minimum = 1e-30

""" Implementation of LASSONET.
    Code adapted from https://github.com/lasso-net/lassonet
    Paper: https://jmlr.org/papers/volume22/20-848/20-848.pdf
"""
def _lambda_seq(start: float, path_multiplier: float):
        while True:
            yield start
            start *= path_multiplier


def soft_threshold(l, x):
    return torch.sign(x) * torch.relu(torch.abs(x) - l)


def sign_binary(x):
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)


def prox(v, u, *, lambda_, lambda_bar, M):
    """
    v has shape (m,) or (m, batches)
    u has shape (k,) or (k, batches)
    supports GPU tensors
    """
    onedim = len(v.shape) == 1
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values

    k, batch = u.shape

    s = torch.arange(k + 1.0).view(-1, 1).to(v)
    zeros = torch.zeros(1, batch).to(u)

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
    )

    norm_v = torch.norm(v, p=2, dim=0)

    # add extra line to avoid 0 division
    norm_v += minimum

    x = F.relu(1 - a_s / norm_v) / (1 + s * M ** 2)

    w = M * x * norm_v
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    idx = torch.sum(lower > w, dim=0).unsqueeze(0)

    x_star = torch.gather(x, 0, idx).view(1, batch)
    w_star = torch.gather(w, 0, idx).view(1, batch)

    beta_star = x_star * v
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star
    #return beta_star.clamp(epsilon, 1-epsilon), theta_star.clamp(epsilon, 1-epsilon)  # modify this to avoid nan.


def inplace_prox(beta, theta, lambda_, lambda_bar, M):
    beta.weight.data, theta.weight.data = prox(
        beta.weight.data, theta.weight.data, lambda_=lambda_, lambda_bar=lambda_bar, M=M
    )

class LassoNet(BaseRanker):
    # Trainable linear model.
    def __init__(self, hparams: Dict[str, Any], hparams_dec: Dict[str, Any]):
        """ hparams contains LassoNet related parameters, hparams_p contains Predictor parameters."""
        super(LassoNet, self).__init__(hparams['train_mode'], hparams['rank_loss'])
        self.deep = StackLinear(hparams_dec['input_dim'], hparams_dec['hidden_dim'], hparams_dec['output_dim'], hparams_dec['num_layers'])
        self.skip = nn.Linear(hparams_dec['input_dim'], hparams_dec['output_dim'], bias=False)
        
        self.gamma = hparams['gamma']
        self.gamma_skip = hparams['gamma_skip']
        self.lambda_ = hparams['lambda_']
        self.M = hparams['M']

        self.automatic_optimization = False
        self.save_hyperparameters() # save hyperparameters in hparams.

    def forward(self, Input):  # L, H -> L,
        assert len(Input.shape) == 2 
        result_skip = self.skip(Input).squeeze(-1)
        result_deep = self.deep(Input).squeeze(-1)
        return result_skip + result_deep
    
    def prox(self, *, lambda_, lambda_bar=0, M=1):
        with torch.no_grad():
            inplace_prox(
                beta=self.skip,
                theta=self.deep.layers[0],
                lambda_=lambda_,
                lambda_bar=lambda_bar,
                M=M,
            )

    def on_train_batch_start(self, batch, batch_idx) -> None:
        if self.selected_count() == 0:   # stop training when selected features are 0
            print(f'Selected feature num: 0, stop training.')
            return -1

    def on_after_backward(self) -> None:
        """ Add this to avoid NAN, INF problem."""
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            self.log.warning(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def training_step(self, batch, batch_idx) :
        if self.global_step % 500 == 0:
            torch.cuda.empty_cache()     # clear cache at regular interval
        optimizer = self.optimizers()
        optimizer.zero_grad()

        if self.train_mode == 'pairwise':
            high_data, low_data, high_label, low_label = batch
            #print(f'Input shape: {high_data.shape}, {low_data.shape}.')
            high_logits = self(high_data)
            low_logits = self(low_data)
            rank_loss = self.rank_loss(high_logits, low_logits, high_label, low_label)
        else:
            data_batch, label_batch = batch
            logits = self(data_batch)
            rank_loss = self.rank_loss(logits, label_batch)/ label_batch.shape[0]   # normalized by doc length

        self.manual_backward(rank_loss)
        optimizer.step()

        if self.lambda_:   
            self.prox(lambda_= self.lambda_ * 1e-3, M = self.M)  # fix learning rate for now

    
    def test_step(self, batch, batch_idx ):
        #_, ndcg = super().test_step(batch, batch_idx)
        test_output = super().test_step(batch, batch_idx)
        l1_loss = self.l1_regularization_skip().item()
        test_output['l1_loss'] = l1_loss
        return test_output


    def test_epoch_end(self, outputs):
        selected = self.selected_count()
        test_result = super().test_epoch_end(outputs)
        L1 = np.mean([output['l1_loss'] for output in outputs])
        self.log('test_reg', round(L1, 3))
        self.log('selected', selected)
        self.log('lambda_', round(self.lambda_, 3))
        test_result[-1].update({'test_reg': round(L1, 3), 'selected': selected, 'lambda_': self.lambda_})
        return test_result
        

    def l2_regularization(self):
        """
        L2 regulatization of the MLP without the first layer
        which is bounded by the skip connection
        """
        ans = 0
        for layer in islice(self.deep.layers, 1, None):
            ans += (
                torch.norm(
                    layer.weight.data,
                    p=2,
                )
                ** 2
            )
        return ans

    def l1_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=1)   # abs value.

    def l2_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2)

    def input_mask(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight.data, p=1, dim=0) != 0
        #with torch.no_grad():
            #return (self.skip.weight.data.abs() > epsilon).sum()


    def selected_count(self):
        return self.input_mask().sum().item()

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}


def compute_feature_importance(model_dir: Path, data_dim: int, path_depth: int):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    ans = torch.full((data_dim,), float('inf')).cuda()
    current = torch.ones(data_dim).int().cuda()
    for i in range(path_depth):
        best_ckpt = model_dir / f'version_{i}' / 'checkpoints/'
        checkpoint = str(list(best_ckpt.glob('*.ckpt'))[0])
        params = str(list(best_ckpt.glob('*.json'))[0])
        if not os.path.isfile(params):
            break
        with open(params, 'r')as f:
            Params = json.load(f)
        lambda_ = Params[0]['lambda_']
        state_dict = torch.load(checkpoint, map_location=map_location)
        skip = state_dict['state_dict']['skip.weight'][0].data.clone()
        #mask = skip != 0
        mask = skip > minimum
        diff = current & ~mask
        ans[diff.nonzero().flatten()] = round(lambda_, 2)
        current &= mask
    return ans


class LassonetPath(object):
    def __init__(self, hparams: Dict[str, Any], hparams_dec: Dict[str, Any]) -> None:
        self.hparams = hparams
        self.hparams_dec = hparams_dec
        self.default_root_dir = Path(hparams['default_root_dir'])
        
    def _run(self, train_loader, valid_loader, test_loader):
        self._train_orig(train_loader, valid_loader, test_loader)
        self._train_path(train_loader, valid_loader, test_loader)

    def _train_orig(self, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):
        self.hparams['lambda_'] = 0
        model = LassoNet(self.hparams, self.hparams_dec).double()
        early_stopping = EarlyStopping(monitor='validation_ndcg@10', mode='max', stopping_threshold=1.0,  patience=10, verbose=True)
        model_checkpoint = ModelCheckpoint(monitor='validation_ndcg@10', mode='max', verbose=True)
        #logger = TensorBoardLogger(self.hparams['model_dir'].parent, name=self.model_name, version='init')
        trainer = pl.Trainer(default_root_dir=self.default_root_dir, strategy='ddp', detect_anomaly=True, gpus=self.hparams['gpu'], max_epochs=20, callbacks=[early_stopping, model_checkpoint], deterministic=True)
        
        trainer.fit(model, train_loader, valid_loader)
        result_init = trainer.test(ckpt_path='best', dataloaders=test_loader)
        best_ckpt = Path(trainer.checkpoint_callback.best_model_path)
        with open(best_ckpt.parent / 'test_results.json', 'w')as f:
            json.dump(model.test_result, f)
        self.result_init = result_init
        self.best_ckpt = best_ckpt

    def _train_path(self, train_loader, valid_loader, test_loader):
        
        # start training path from begining.
        if self.hparams['lambda_start']:
            lambda_start = self.hparams['lambda_start']
        else:
            lambda_start = self.result_init[-1]['test_reg']
        
        # continue path training from version {recover_version}.
        if self.hparams['recover_version'] > 0:
            checkpoint_dir = self.default_root_dir / f"version_{self.hparams['recover_version']}" / 'checkpoints'
            self.best_ckpt = str(list(checkpoint_dir.glob('*.ckpt'))[0])
            with open( checkpoint_dir / 'test_results.json', 'r')as f:
                test_results = json.load(f)
            lambda_start = test_results[-1]['lambda_'] * self.hparams['path_multiplier']

        lambda_sequence = _lambda_seq(lambda_start, self.hparams['path_multiplier'])
        
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'
        
        max_try = max(0, self.hparams['recover_version'] + 1)
        for lambda_ in lambda_sequence:
            if max_try >self.hparams['path_depth']:
                break
           
            else:
                print(f'Lambda: {lambda_}...')
                self.hparams['lambda_'] = lambda_
            
                early_stopping = EarlyStopping(monitor='validation_ndcg@10', mode='max', stopping_threshold=1.0,  patience=self.hparams['path_patience'], verbose=True)
                model_checkpoint = ModelCheckpoint(monitor='validation_ndcg@10', mode='max', verbose=True)
                trainer = pl.Trainer(default_root_dir=self.default_root_dir, strategy='ddp', detect_anomaly=True, gpus=self.hparams['gpu'], max_epochs=self.hparams['path_max_epoch'], callbacks=[early_stopping, model_checkpoint], deterministic=True)

                model = LassoNet(self.hparams, self.hparams_dec).double()
                print(f'Load state dict from {self.best_ckpt}...\n')
                state_dict = torch.load(self.best_ckpt, map_location=map_location)
                model.load_state_dict(state_dict['state_dict'])
                trainer.fit(model, train_loader, valid_loader)
                result_path = trainer.test(ckpt_path='best', dataloaders=test_loader)
                best_ckpt = Path(trainer.checkpoint_callback.best_model_path)
                with open(best_ckpt.parent / 'test_results.json', 'w')as f:
                    json.dump(model.test_result, f)
                self.best_ckpt = best_ckpt

                max_try += 1

    def load_state_dicts(self, ckpt: Path):
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'
        model = LassoNet(self.hparams, self.hparams_dec).double()
        print(f'Load state dict from {ckpt}...\n')
        state_dict = torch.load(ckpt, map_location=map_location)
        model.load_state_dict(state_dict['state_dict'])
        return model

    def explain(self, saveto: Path=False):
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
            device = torch.device('cuda')
        else:
            map_location='cpu'
            device = torch.device('cuda')
        ans = torch.full((self.hparams_dec['input_dim'],), float('inf'))
        current = torch.ones(self.hparams_dec['input_dim']).int().to(device)
        for i in range(self.hparams['path_depth']):
            best_ckpt = self.default_root_dir / 'lightning_logs' / f'version_{i}' / 'checkpoints/'
            checkpoint = str(list(best_ckpt.glob('*.ckpt'))[0])
            params = str(list(best_ckpt.glob('*.json'))[0])
            if not os.path.isfile(params):
                break
            with open(params, 'r')as f:
                Params = json.load(f)
            lambda_ = Params[-1]['lambda_']
            state_dict = torch.load(checkpoint, map_location=map_location)
            skip = state_dict['state_dict']['skip.weight'][0].data.clone()
            mask = skip != 0
            #mask = skip > 0
            diff = current & ~mask
            ans[diff.nonzero().flatten()] = round(lambda_, 2)
            current &= mask
        #return ans

        indices = torch.argsort(ans, descending=True).data.cpu().tolist()
        if saveto:
            FI_path = saveto / 'feature_importances.json'
            with open(FI_path, 'w')as f:
                json.dump(indices, f)
        print(f'Save feature importance to {FI_path}.')

        return indices




        
