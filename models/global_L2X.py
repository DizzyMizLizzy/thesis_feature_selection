from typing import Any, Dict, List, Tuple
import torch
from networks import DeepSet
from base_ranker import BaseRanker
import json
from pathlib import Path


class ConcreteGlobalSelector(BaseRanker):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        super(ConcreteGlobalSelector, self).__init__(hparams['train_mode'], hparams['rank_loss'])
        self.input_dim = hparams['input_dim']
        self.output_dim = hparams['output_dim']
        self.predictor_hidden = hparams['predictor_hidden']
        self.num_layers = hparams['num_layers']
        self.start_temp = hparams['start_temp']
        self.temp = hparams['start_temp']
        self.min_temp = hparams['min_temp']
        self.k = int(self.input_dim * hparams['feature_ratio'])
        self.eps = torch.finfo(torch.float32).eps
        self.supervise = hparams['supervise']
        
        if hparams['predictor'] == 'linear':
            self.predictor = DeepSet(self.input_dim, self.predictor_hidden, self.output_dim, self.num_layers)
        else:
            raise ValueError(f"Invalid predictor: {hparams['predictor']}.")
        
        # Initialize selecting layer.
        logits = torch.empty((1, self.input_dim), dtype=torch.float64, requires_grad=True)
        torch.nn.init.xavier_uniform_(logits)
        self.logits = torch.nn.Parameter(logits)
        self.save_hyperparameters() # save hyperparameters in hparams.

    def forward(self, x):
        mask, _ = self._get_mask()
        x_hat = x * mask
        pred = self.predictor(x_hat).squeeze(-1)
        return pred
    
    def _temp(self):
        """ Temprature Annealing Schedule"""
        #print(f'current epoch: {self.current_epoch}.')
        T = self.start_temp * (0.01 / self.start_temp) ** (self.current_epoch  / 50)  # suppose 10 epochs for now.
        return max(0.01, T)
    
    def on_train_epoch_end(self):
        self.temp = self._temp()

    def _get_mask(self):
        if self.training:
            B, H = self.logits.shape
            logits = self.logits.unsqueeze(1)   # B, 1, H
            uniform = (self.eps - 1.0) * torch.rand((B, self.k, H)) + 1.0 - self.eps
            gumbel = -torch.log(-torch.log(uniform)).to(logits.device)
            noisy_logits = (logits + gumbel) / self.temp
            samples = torch.softmax(noisy_logits, -1)
            samples = torch.max(samples, dim=1)[0]   # B, k, H -> B, H
            _, topk_indices = samples.topk(self.k, dim=-1)
            k_hot = torch.zeros_like(samples).scatter_(1, topk_indices, 1)  
            return (k_hot - samples).detach() + samples, topk_indices
        else:
            _, topk_indices = self.logits.topk(self.k, dim=-1)
            k_hot = torch.zeros_like(self.logits).scatter_(1, topk_indices, 1)
            return k_hot, topk_indices


    def explain_step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mask, indices = self._get_mask()
        return mask, indices.squeeze(0)
    
    def explain(self, test_loader, saveto: Path=False) -> Dict[str, List]:
        """Compute the selected features, global feature selection has no involement of input data. 
            Args:
                nonrepeat: if yes, select features without repeatition. 
            Return:
                feature indices selected by the selected layer.
         """
        self.eval()
        with torch.no_grad():
            _, indices = self.explain_step()
        features_sorted = indices.tolist()
        if saveto:
            with open(saveto / 'feature_importances.json', 'w')as f:
                json.dump(features_sorted, f)
            print(f'Saved ranked features to {saveto}.')
        return {'feature_importances': features_sorted}
    

        
    


