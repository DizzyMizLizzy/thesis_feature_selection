from typing import Any, Dict, Iterable, List
import torch
import json
from networks import DeepSet
from base_ranker import BaseRanker
from tqdm import tqdm
from collections import Counter
from pathlib import Path


class L2XRanker(BaseRanker):
    """ Implementation of L2X for LTR task.
        paper: https://arxiv.org/pdf/1802.07814.pdf
        code partially adapted from: https://github.com/Jianbo-Lab/L2X
    """
    def __init__(self, hparams: Dict[str, Any]):
        super(L2XRanker, self).__init__(hparams['train_mode'], hparams['rank_loss'])

        self.actor = DeepSet(hparams['input_dim'], hparams['hidden_dim'], hparams['input_dim'], hparams['num_layers']-1)
        self.critic = DeepSet(hparams['input_dim'], hparams['hidden_dim'], hparams['output_dim'], hparams['num_layers']) # predictor
        self.k = int(hparams['input_dim'] * hparams['feature_ratio'])
        self.tau = hparams['temperature']   # 0.1 by default.
        self.eps = torch.finfo(torch.float32).eps
        self.straight_through = hparams['straight_through']
        self.critic_only = False
        self.save_hyperparameters() # save hyperparameters in hparams.

    def _get_mask(self, Input, k_, tau):
        """ Generate masks via Gumble-softmax sampling, w/o straight-through trick.
            or discrete argmax in Inference. """
        logits = self.actor(Input)  # B, H
        B, H = logits.shape
        if self.training:
            logits = logits.unsqueeze(1)   # B, 1, H
            uniform = (self.eps - 1.0) * torch.rand((B, k_, H)) + 1.0 - self.eps
            gumbel = -torch.log(-torch.log(uniform)).to(logits.device)
            noisy_logits = (logits + gumbel) / tau
            samples = torch.softmax(noisy_logits, -1)
            samples = torch.max(samples, dim=1)[0]   # B, num_top_group, num_group -> B, num_group
            _, topk_indices = samples.topk(k_, dim=-1)

            if self.straight_through:
                k_hot = torch.zeros_like(samples).scatter_(1, topk_indices, 1)  
                return (k_hot -samples).detach() + samples, topk_indices
            else:
                return samples, topk_indices
        else:
            _, topk_indices = torch.topk(logits, k_, dim=-1)
            k_hot = torch.zeros_like(logits).scatter_(1, topk_indices, 1)
            return k_hot, topk_indices

    def forward(self, Input, mask=True):
        assert len(Input.shape) == 2 
        mask, _ = self._get_mask(Input, self.k, self.tau)
        X_masked = Input * mask
        out = self.critic(X_masked).squeeze(-1)
        return out

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        data_batch, label_batch = batch
        if len(data_batch.shape) > 2:
            data_batch = data_batch.squeeze(0)
            label_batch = label_batch.squeeze(0)
        
        if self.critic_only:
            logits = self.critic(data_batch).squeeze(-1)
        else:
            logits = self(data_batch)
        if isinstance(logits, (list, tuple)):
            logits = logits[0].detach().data
       
        return {'preds':logits, 'labels':label_batch}

    def explain_step(self, Input):
        if len(Input.shape) > 2:
            Input = Input.squeeze(0).to(self.device)
        mask, indices = self._get_mask(Input, self.k, self.tau)
        return mask, indices
    
    def explain(self, X_iter: Iterable, saveto: Path = False, fname='feature_importances.json') -> Dict[str, List]:
        """Compute the selected features for input data, e.g., test dataset. 
            Return:
                feature indices sorted by their selected frequency.
         """
        self.eval()
        Indices = []
        with torch.no_grad():
            for batch in tqdm(X_iter):
                data, _ = batch
                _, indices = self.explain_step(data)   
                Indices += indices.flatten().detach().cpu().tolist() 
        counts = Counter(Indices)
        features_sorted = sorted(counts, key=counts.get, reverse=True)
        if saveto:
            with open(saveto / fname, 'w')as f:
                json.dump(features_sorted, f)
        return {'feature_importances': features_sorted}

