import torch
from pytorch_tabnet import tab_network
from typing import Iterable, List, Dict, Any, Tuple
import json, pickle
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from base_ranker import BaseRanker


class TabnetRanker(BaseRanker):
    """ List-wise ranking model using tabnet architecture"""
    def __init__(self, hparams: Dict):
        super(TabnetRanker, self).__init__(hparams['train_mode'], hparams['rank_loss'])
        self.input_dim = hparams['input_dim']
        self.output_dim = hparams['output_dim']
        self.n_d = hparams['n_d']
        self.n_a = hparams['n_a']
        self.n_steps = hparams['n_steps']
        self.gamma = hparams['gamma']   # 1.3 by default.
        self.n_independent = hparams['n_independent']  # num of independant layers in feature transformer.
        self.n_shared = hparams['n_shared']  # num of shared layers in feature transformer.
        self.virtual_batch_size = hparams['virtual_batch_size']
        self.momentum = hparams['momentum']  # 0.02
        self.mask_type = hparams['mask_type']  #sparsemax
        #self.rank_loss = metrics.get_loss(hparams['rank_loss'])   # loss function, by default softmax_cross_entropy.
        self.lambda_sparse = hparams['lambda_sparse']  # default = 0.001
        self.epsilon = torch.finfo(torch.float32).eps

        self._set_network()
        self.save_hyperparameters() # save hyperparameters in hparams.

    
    def _set_network(self) -> None:
        """ Define Tabnet model.
            network output: (final prediction, mask_loss)
        """
        self.network = tab_network.TabNetNoEmbeddings(self.input_dim, self.output_dim, n_d=self.n_d, n_a=self.n_a,
            n_steps=self.n_steps, gamma=self.gamma, n_independent=self.n_independent, n_shared=self.n_shared,
            epsilon=self.epsilon, virtual_batch_size=self.virtual_batch_size, momentum=self.momentum, mask_type=self.mask_type)

    def forward(self, Input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function of tabnet model.
            Args:
                Input: a batch of Input, Input.shape == batch_size, num_features.
            Return:
                - score logits of tabnet network, without normalization.
                - entropy loss of mask probs, to enforce sparsity.
        """
        assert len(Input.shape) == 2 
        out, M_loss = self.network(Input)
        out = out.squeeze(-1) # L, 1 -> L,
        return out, M_loss

    def predict(self, X) -> Tuple[torch.Tensor, torch.Tensor]:
        """call forward function, with input shape check."""
        if len(X.shape) > 2:  # valid, test data are read by query. 
            X = X.squeeze(0)
        return self.forward(X)

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """Training a batch of input data. Minimizing ranking loss and entropy loss.
            Args
                batch of data: (data, label).
            Return
                loss = rank_loss + lambda_sparse * M_loss.
        """
        if self.train_mode == 'pairwise':
            high_data, low_data, high_label, low_label = batch
            #print(f'Input shape: {high_data.shape}, {low_data.shape}.')
            high_logits, M_high= self(high_data)
            low_logits, M_low = self(low_data)
            rank_loss = self.rank_loss(high_logits, low_logits, high_label, low_label)
            loss = rank_loss - self.lambda_sparse * (M_high + M_low)
        else:
            data_batch, label_batch = batch
            assert len(data_batch.shape) == 2
            batch_size, _ = data_batch.shape
            logits, M_loss = self(data_batch)
            rank_loss = self.rank_loss(logits, label_batch) / batch_size
            loss = rank_loss - self.lambda_sparse * M_loss
            self.log('M_loss', M_loss.data, on_step=True, on_epoch=True)
        
        self.log('rank_loss', rank_loss.data, on_step=True, on_epoch=True)    
        self.log('train_loss', loss.data, on_step=True, on_epoch=True)
        return {'loss': loss}

    def explain_step(self, Input: torch.Tensor) -> Tuple[np.array, Dict[str, np.array]]:
        """ Compute the aggregated mask and step-wise masks for batch Input.
            Args: 
                Input: Input.shape == L, H
            Return:
                -mask_aggregated:  np.array
                -mask_steps:  dict object
        """
        if len(Input.shape) > 2:
            Input = Input.squeeze(0).to(self.device)
        with torch.no_grad():
            M_explain, masks = self.network.forward_masks(Input)
        M_explain = M_explain.detach().cpu().numpy()
        for k, v in masks.items():
            masks[k] = v.detach().cpu().numpy()   
        return M_explain, masks

    def explain(self, X_iter: Iterable, normalize:bool=True, saveto: Path=False, fname: str= 'feature_importances.json') -> Dict[str, Any]:
        """Compute the aggregated masks, step-wise masks, the number of selected features per instance.
            Args:
                Input data: dataLoader object.
            Return:
                - mask_aggregated: np.array
                - mask_steps: dict object
                - selected_num: np.array

        """
        res_explain = []
        for batch_nb, batch in tqdm(enumerate(X_iter)):
            data, _ = batch
            mask_agg, masks = self.explain_step(data)        
            res_explain.append(mask_agg)
            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])
        res_explain = np.vstack(res_explain)
        if normalize:
            res_explain /= np.sum(res_explain, axis=1)[:, None]
        selected_num = (res_explain > 0).sum(-1).tolist()
        selected_avg = np.mean(selected_num)
        selected_std = np.std(selected_num)
        selected_num.append((selected_avg, selected_std))  # the last element is the (avg, std) tuple.

        feature_sum = res_explain.sum(0)
        indices = np.argsort(-feature_sum).tolist()
        if saveto:
            if isinstance(saveto, str):
                saveto = Path(saveto)

            with open(saveto / fname, 'w')as f:
                json.dump(indices, f)
            #with open(saveto / 'selected_statistic.json', 'w')as f:
            #    json.dump(selected_num, f)
            #with open(saveto / 'explains.pkl', 'wb')as f:
            #    pickle.dump(res_explain, f)
            #with open(saveto / 'masks.pkl', 'wb')as f:
            #    pickle.dump(res_masks, f)
            print(f'Saved ranked feature indices, selected nums, explains and masks to {saveto}.')
        return {'mask_aggregated': res_explain, 'mask_steps': res_masks, 'selected_num':selected_num, 'ranked_indices': indices}

    def visualize(self, importance:np.array, masks: Dict[int, np.array], num: int, saveto: str):
        """ Visualize feature importance and masks from all steps. choose to show {num} instances."""
        assert importance.shape[0] >= num
        step = len(masks.keys())
        fig, axs = plt.subplots(1, step+1, figsize=(30,30))
        axs[0].imshow(importance[:num])
        axs[0].set_title(f"mask aggregated")
        for i in range(1,step+1):
            axs[i].imshow(masks[i-1][:num])
            axs[i].set_title(f"mask {i}")
        plt.savefig(saveto)
