import torch
from typing import List, Dict, Any, Tuple, Iterable
import metrics
from networks import DeepSet
from base_ranker import BaseRanker
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path


class InvaseRanker(BaseRanker):
    """ INVASE: Instance-wise variable selection using NN, implementation for LTR
        paper: https://openreview.net/pdf?id=BJg_roAcK7
        code partially adapted from: https://github.com/jsyoon0823/INVASE/blob/master/invase.py
    """
    def __init__(self, hparams: Dict[str, Any]) -> None:
        super(InvaseRanker, self).__init__(hparams['train_mode'], hparams['rank_loss'])
        self.input_dim = hparams['input_dim']
        self.output_dim = hparams['output_dim']
        self.actor = DeepSet(hparams['input_dim'], hparams['hidden_dim'], hparams['input_dim'], hparams['num_layers'])
        self.critic = DeepSet(hparams['input_dim'], hparams['hidden_dim'], hparams['output_dim'], hparams['num_layers']) # predictor
        self.baseline = DeepSet(hparams['input_dim'], hparams['hidden_dim'], hparams['output_dim'], hparams['num_layers']) # baseline
        #self.rank_loss = metrics.get_loss(hparams['rank_loss'])
        self.lamda = hparams['lamda']   # for sparsity loss
        self.automatic_optimization = False
        
        self.save_hyperparameters() # save hyperparameters in hparams.

    
    def _actor_out(self, X):
        logits = self.actor(X) 
        return torch.sigmoid(logits)  # generate selector probs
    
    def _critic_out(self, X, mask):
        X_masked = X * mask
        logits = self.critic(X_masked).squeeze(-1) # do softmax in loss function
        return logits
    
    def _baseline_out(self, X):
        logits = self.baseline(X).squeeze(-1)
        return logits

    def _actor_loss(self, y_true, y_pred):
        """Args:
        Copied from https://github.com/jsyoon0823/INVASE/blob/master/invase.py
      - y_true:
        - actor_out: actor output after sampling
        - critic_out: critic output 
        - baseline_out: baseline output (only for invase)
      - y_pred: output of the actor network, already sigmoid
        
        Returns:
         - loss: actor loss
        """   
        actor_out = y_true[:, :self.input_dim]
        critic_out = y_true[:, self.input_dim:(self.input_dim+self.output_dim)]
        # Baseline output
        baseline_out = y_true[:, (self.input_dim+self.output_dim):(self.input_dim+2*self.output_dim)]
        # Ground truth label
        y_out = y_true[:, (self.input_dim+2*self.output_dim):]   

        critic_prob = torch.softmax(critic_out, -1)
        critic_loss = -torch.sum(y_out*torch.log(critic_prob), -1)
        
        baseline_prob = torch.softmax(baseline_out, -1)
        baseline_loss = -torch.sum(y_out*torch.log(baseline_prob), -1)
        Reward = -(critic_loss - baseline_loss)   # original paper.
        #Reward = torch.abs(critic_loss - baseline_loss)   # modify it since the direction keeps change.

        custom_actor_loss = Reward * torch.sum(actor_out * torch.log(y_pred) + (1-actor_out) * torch.log(1-y_pred), axis = 1) - \
                            self.lamda * torch.mean(y_pred, axis = 1)
        
        # custom actor loss
        custom_actor_loss = torch.mean(-custom_actor_loss)
        return custom_actor_loss

    def predict(self, X):
        """ Inference, select feature iff mask >= 0.5
            Return prediction without activation, hard mask.
        """
        mask_prob = self._actor_out(X)
        mask_hard = torch.round(mask_prob)
        pred = self._critic_out(X, mask_hard)
        return pred, mask_hard

    def training_step(self, batch, batch_idx):
        opt_critic, opt_baseline, opt_actor = self.optimizers()
        opt_critic.zero_grad()
        opt_baseline.zero_grad()
        opt_actor.zero_grad()

        if self.train_mode == 'pairwise':
            high_data, low_data, high_label, low_label = batch
            #print(f'Input shape: {high_data.shape}, {low_data.shape}.')
            mask_probs_high = self._actor_out(high_data)
            mask_hard_high = torch.bernoulli(mask_probs_high)
            Pred_critic_high = self._critic_out(high_data, mask_hard_high)
            Pred_baseline_high = self._baseline_out(high_data)

            mask_probs_low = self._actor_out(low_data)
            mask_hard_low = torch.bernoulli(mask_probs_low)
            Pred_critic_low = self._critic_out(low_data, mask_hard_low)
            Pred_baseline_low = self._baseline_out(low_data)
      
            critic_loss = self.rank_loss(Pred_critic_high, Pred_critic_low, high_label, low_label)
            baseline_loss = self.rank_loss(Pred_baseline_high, Pred_baseline_low, high_label, low_label)
            
        else:
            X, Y = batch
            mask_probs = self._actor_out(X)
            mask_hard = torch.bernoulli(mask_probs)

            Pred_critic = self._critic_out(X, mask_hard)
            Pred_baseline = self._baseline_out(X)
            baseline_loss = self.rank_loss(Pred_baseline, Y)
            critic_loss = self.rank_loss(Pred_critic, Y)

        self.manual_backward(baseline_loss)
        opt_baseline.step()
        self.manual_backward(critic_loss, retain_graph=True)   # retain the graph here, for training actor next.
        opt_critic.step()

        # train actor now
        Pred_critic = self._critic_out(X, mask_hard)
        Pred_baseline = self._baseline_out(X)
        actor_true = torch.cat((mask_hard, Pred_critic.unsqueeze(-1), Pred_baseline.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1)
        actor_loss = self._actor_loss(actor_true, mask_probs)
        self.manual_backward(actor_loss)
        opt_actor.step()

        #print(f'Training selected feature number: {(mask_hard>0).sum()/Y.shape[0]}.')
        self.log_dict({'baseline_loss': baseline_loss.data, 'critic_loss': critic_loss.data, 'actor_loss': actor_loss.data})

    def validation_step(self, batch, batch_idx) -> Tuple[float, float]:
        data_batch, label_batch = batch
        if len(data_batch.shape) > 2:
            data_batch, label_batch = data_batch.squeeze(0), label_batch.squeeze(0)
        mask_probs = self._actor_out(data_batch)
        mask_hard = torch.round(mask_probs)
        #print(f'Valid selected number: {(mask_hard>0).sum()/label_batch.shape[0]}\n')
        pred_critic = self._critic_out(data_batch, mask_hard).detach().cpu().numpy()
        pred_baseline = self._baseline_out(data_batch).detach().cpu().numpy()
        label_batch = label_batch.cpu().numpy()
        ndcg_critic = metrics.list_ndcg(pred_critic, label_batch, self.early_signal)
        ndcg_baseline = metrics.list_ndcg(pred_baseline, label_batch, self.early_signal)
        return ndcg_critic, ndcg_baseline
    
    def test_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float]]:
        data_batch, label_batch = batch
        if len(data_batch.shape) > 2:
            data_batch, label_batch = data_batch.squeeze(0), label_batch.squeeze(0)
        mask_probs = self._actor_out(data_batch)
        mask_hard = torch.round(mask_probs)
        #print(f'Test selected number: {(mask_hard>0).sum()/label_batch.shape[0]}\n')
        
        label_batch = label_batch.cpu().numpy()
        pred_critic = self._critic_out(data_batch, mask_hard).detach().cpu().numpy()
        pred_baseline = self._baseline_out(data_batch).detach().cpu().numpy()
        ndcg_critic = [metrics.list_ndcg(pred_critic, label_batch, i) for i in self.ndcg_truncate]
        ndcg_baseline = [metrics.list_ndcg(pred_baseline, label_batch, i) for i in self.ndcg_truncate]
        return pred_critic, pred_baseline, ndcg_critic, ndcg_baseline
    
    def validation_epoch_end(self, outputs) -> None:
        valid_ndcg = np.mean([output[0].cpu().numpy() for output in outputs])
        self.log(f'validation_ndcg@{self.early_signal}', valid_ndcg)
    
    def test_epoch_end(self, outputs) -> Dict[str, float]:
        test_result = torch.tensor([output[2] for output in outputs]).cpu().list()
        avg_result = {}
        ndcg_critic = [np.mean([output[2][i].cpu().numpy() for output in outputs]) for i, _ in enumerate(self.ndcg_truncate)]
        ndcg_baseline = [np.mean([output[3][i].cpu().numpy() for output in outputs]) for i, _ in enumerate(self.ndcg_truncate)]
        for i, k in enumerate(self.ndcg_truncate):
            key = f'test_ndcg@{k}'
            key_base = f'baseline_ndcg@{k}'
            avg_result[key] = ndcg_critic[i]
            avg_result[key_base] = ndcg_baseline[i]
            self.log(f'test_ndcg@{k}', ndcg_critic[i])
            self.log(f'baseline_ndcg@{k}', ndcg_baseline[i])
        test_result.append(avg_result)
        self.test_result = test_result
        return test_result

    def explain_step(self, Input: torch.Tensor) -> Tuple[np.array, Dict[str, np.array]]:
        """ Compute mask selected by actor model.
            Args: 
                Input: Input.shape == L, H
            Return:
                mask_hard:  0-1 array.
        """
        if len(Input.shape) > 2:
            Input = Input.squeeze(0).to(self.device)
        with torch.no_grad():
            mask_probs = self._actor_out(Input)
            mask_hard = torch.round(mask_probs).detach()
        return mask_hard, 0

    def explain(self, X_iter: Iterable, saveto: Path=False) -> Dict[str, Any]:
        """Compute the aggregated masks, step-wise masks, the number of selected features per instance.
            Args:
                Input data: dataLoader object.
            Return:
                - mask_aggregated: np.array
                - mask_steps: dict object
                - selected_num: np.array

        """
        Masks = []
        for batch_nb, batch in tqdm(enumerate(X_iter)):
            data, _ = batch
            masks, _ = self.explain_step(data)        
            Masks.append(masks.cpu().numpy())
        Masks = np.vstack(Masks)
       
        selected_num = Masks.sum(-1).tolist()
        selected_avg = np.mean(selected_num)
        selected_std = np.std(selected_num)
        selected_num.append((selected_avg, selected_std))  # the last element is the (avg, std) tuple.

        feature_sum = Masks.sum(0)
        indices = np.argsort(-feature_sum).tolist()
        if saveto:
            with open(saveto / 'feature_importances.json', 'w')as f:
                json.dump(indices, f)
            with open(saveto / 'selected_statistic.json', 'w')as f:
                json.dump(selected_num, f)
            print(f'Saved ranked feature indices, selected nums, explains and masks to {saveto}.')
        
        return {'selected_num':selected_num, 'ranked_indices': indices}


    def configure_optimizers(self):
        optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3, weight_decay=1e-4)    
        optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-4) 
        optimizer_baseline = torch.optim.Adam(self.baseline.parameters(), lr=1e-3, weight_decay=1e-4) 
        return optimizer_critic, optimizer_baseline, optimizer_actor

       

    
