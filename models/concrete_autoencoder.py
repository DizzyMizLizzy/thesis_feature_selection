import torch
from typing import Dict, List, Any, Tuple
from networks import DeepSet
import numpy as np
from base_ranker import BaseRanker
import json
from pathlib import Path


""" Implementation of the paper ```Concrete AutoEncoders: Differentiable Feature Selection and Reconstruction```
    paper: http://proceedings.mlr.press/v97/balin19a/balin19a.pdf
    code: https://github.com/mfbalin/Concrete-Autoencoders
    This implementation supports both supervised(Y label) and unsupervised(reconstruction) training.
"""
class ConcreteAutoEncoder(BaseRanker):
    def __init__(self, hparams: Dict[str, Any]):
        super(ConcreteAutoEncoder, self).__init__(hparams['train_mode'], hparams['rank_loss'])
        self.input_dim = hparams['input_dim']
        self.output_dim = hparams['output_dim']
        self.predictor_hidden = hparams['predictor_hidden']
        self.num_layers = hparams['num_layers']
        self.start_temp = hparams['start_temp']
        self.min_temp = hparams['min_temp']
        self.k = int(self.input_dim * hparams['feature_ratio'])
        self.eps = torch.finfo(torch.float32).eps
        self.supervise = hparams['supervise']      # we always use supervise training in this work.
        
        if self.supervise:  # optimize reconstruction loss
            print(f'Y label training, supervised.')
            #self.rank_loss = metrics.get_loss(hparams['rank_loss'])
            if hparams['predictor'] == 'linear':
                self.predictor = DeepSet(self.k, self.predictor_hidden, self.output_dim, self.num_layers)
            else:
                raise ValueError(f"Invalid predictor: {hparams['predictor']}.")
        else:
            print(f'AutoEncoder training, unsupervised.')
            self.mse_loss = torch.nn.MSELoss()
            if hparams['predictor'] == 'linear':
                self.predictor = DeepSet(self.k, self.predictor_hidden, self.input_dim, self.num_layers)  # B, H -> B, H
            else:
                raise ValueError(f"Invalid predictor: {hparams['predictor']}.")
        
        # Initialize selecting layer.
        logits = torch.empty((self.k, self.input_dim), dtype=torch.float64, requires_grad=True)
        torch.nn.init.xavier_uniform_(logits)
        self.logits = torch.nn.Parameter(logits)

        self.save_hyperparameters() # save hyperparameters in hparams.


    def _get_mask(self) -> torch.Tensor:
        if self.training:
            uniform = (self.eps - 1.0) * torch.rand(self.logits.shape) + 1.0 - self.eps  # pytorch rand-> [0, 1), -eps to avoid 1
            gumbel = -torch.log(-torch.log(uniform)).to(self.logits.device)
            temp = self._temp()
            noisy_logits = (self.logits + gumbel) / temp
            samples = torch.softmax(noisy_logits, -1)
            return samples, None    # K, H
        else:
            indices = torch.argmax(self.logits, -1)   # the original code, might be repetitive selection
            k_hot = torch.nn.functional.one_hot(indices, self.logits.shape[1]).double() # Long to double
            return k_hot, indices

    def _get_mask_nonrepeat(self):
        """ The original paper selected the highest feature from each selector neuron, can't guarantee no repeatition.
            Here we choose the max prob from all selector sample neurons for each feature first and then take argmax."""
        #indices = torch.argmax(self.logits, -1).data  # original proposal
        indices = torch.max(self.logits, dim=0)[0]   # k, L -> L, 
        indices = torch.topk(indices, self.k, sorted=True).indices.data
        mask = torch.nn.functional.one_hot(indices, self.logits.shape[-1])
        return mask, indices

    def _temp(self):
        """ Temprature Annealing Schedule"""
        T = self.start_temp * (self.min_temp / self.start_temp) ** (self.current_epoch / 100)  # suppose 100 epochs for now.
        return max(0.01, T)

    def forward(self, X):
        soft_mask = self._get_mask()[0]
        selected = torch.mm(X, soft_mask.transpose(1, 0))  # L, H * H, K -> L, K  
        output = self.predictor(selected).squeeze(-1)   # L, K -> L, H or L, 
        return output

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:  
        if self.supervise:
            loss = super(ConcreteAutoEncoder, self).training_step(batch, batch_idx)
        else:
            data_batch, _ = batch
            logits= self(data_batch)   #  L, H -> L, H, only applies to 1 batch size. or L, 
            loss = self.mse_loss(logits, data_batch)
            self.log('mse_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]: 
        if self.supervise:
            return super(ConcreteAutoEncoder, self).validation_step(batch, batch_idx)
        else:
            data_batch, _ = batch   #x.shape: B, L, H    y.shape: B, L
            if len(data_batch.shape) > 2:
                data_batch = data_batch.squeeze(0)
            logits = self(data_batch)
            valid_loss = self.mse_loss(logits, data_batch).data
            return {'valid_loss':valid_loss}

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        if self.supervise:
            return super(ConcreteAutoEncoder, self).test_step(batch, batch_idx)
        else:
            data_batch, _ = batch   # x.shape: B, L, H    y.shape: B, L
            if len(data_batch.shape) > 2:
                data_batch = data_batch.squeeze(0)
            logits = self(data_batch)
            loss = self.mse_loss(logits, data_batch).data
            return {'mse_loss':loss}

    def validation_epoch_end(self, outputs) -> None:
        if self.supervise:
            return super(ConcreteAutoEncoder, self).validation_epoch_end(outputs)    
        else:
            avg = np.mean([output['mse_loss'] for output in outputs])
            self.log("validation_mse", avg)
            return avg
    
    def test_epoch_end(self, outputs) -> Dict[str, float]:
        if self.supervise:
            return super(ConcreteAutoEncoder, self).test_epoch_end(outputs) 
        else:
            avg = np.mean([output['mse_loss'] for output in outputs])
            self.log("test_mse", avg)
            return {'test_mse': avg} 
    
    def explain_step(self, nonrepeat=True) -> Tuple[torch.Tensor, torch.Tensor]:
        if nonrepeat:
            mask, indices = self._get_mask_nonrepeat()
        else:
            mask, indices = self._get_mask()
        return mask, indices
    
    def explain(self, nonrepeat:bool=False, saveto: Path=False) -> Dict[str, List]:
        """Compute the selected features, global feature selection has no involement of input data. 
            Args:
                nonrepeat: if yes, select features without repeatition. 
            Return:
                feature indices selected by the selected layer.
         """
        self.eval()
        with torch.no_grad():
            _, indices = self.explain_step(nonrepeat=nonrepeat)
        features_sorted = list(set(indices.tolist()))
        if saveto:
            if nonrepeat:
                save_file = 'feature_importances.json'
            else:
                save_file = 'feature_importances_repeat.json'
            with open(saveto / save_file, 'w')as f:
                json.dump(features_sorted, f)
            print(f'Saved ranked features to {saveto}/{save_file}.')
        return {'feature_importances': features_sorted}
    

   