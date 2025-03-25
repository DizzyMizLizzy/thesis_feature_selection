from typing import Any, Dict
import torch
import pytorch_lightning as pl
import torch.nn as nn
from itertools import product
from models import SimpleNet, SelectorPredictor


class NonLeakingSelector(pl.LightningModule):
    # SUWR, sequential feature selection and prediction with stop signal.
    def __init__(self, hparams: Dict[str, Any]) -> None:
        super(NonLeakingSelector, self).__init__()
        """
        hparams: 
            # hyperparameters for the model definition. Defined in models.py
            model_type: simple or selector_predictor. We use simple. 
            input_dim: original input dimension.
            output_dim: output label dimension. 1 for MSE.
            hidden_dim: hidden dimension.
            num_layers: number of hidden layers.

            # hyperparameters for the training.
            task: classification or regression.
            loss_func: loss function for training.
            eval_func: evaluation function for validation, e.g., auroc.
            valid_metric_func: metric function for validation, e.g., accuracy.

            # hyperparameters for the method.
            max_budget: maximum budget for feature selection. It can be a float (ratio) or int.
            step_size: number of features selected per step. By default, it is 1.
            lamda: sparsity weight for training.
            exploration_epochs: the first # epochs for full selection exploration.
            exploration: exploration factor for selection prob. Starts from 1, and decreases to 0.05. Meaning from random selection to relying on selection probs.
            exploration_stop: exploration factor for stop prob. Starts from 1, and decreases to 0.1. Meaning equal stop prob at each step, to relying on stop probs.
            select_patch: select a patch of features around the selected feature. 1/0. For image data, it is better to select a patch of pixels.
        
        """
        
        self.model_type = hparams['model_type']
        self.add_step = hparams['add_step']  # add step t as part of the input. 1/0. No big difference for experiments.
        self.input_dim = hparams['input_dim']
        self.output_dim = hparams['output_dim']
        self.predictor_hidden = hparams['hidden_dim']
        self.num_layers = hparams['num_layers']
        
        if hparams['model_type'] == 'simple':
            print("\n===================== Init simple model =====================\n")
            self.model = SimpleNet(hparams['input_dim'], hparams['hidden_dim'], hparams['output_dim'], hparams['num_layers'], hparams['add_step'])

        elif hparams['model_type'] == 'selector_predictor':
            print("\n===================== Init selector predictor model =====================\n")
            self.model = SelectorPredictor(hparams['input_dim'], hparams['hidden_dim'], hparams['output_dim'], hparams['num_layers'], hparams['add_step'])
        else:
            raise ValueError(f"Model {hparams['model']} not supported.")
        
        self.task = hparams['task']  # classification or regression.
        self.loss_func = hparams['loss_func']
        self.eval_func = hparams['eval_func']
        self.valid_metric_func = hparams['valid_metric_func']  

        if hparams['max_budget'] < 1:
            self.max_budget = max(int(self.input_dim * hparams['max_budget']), 1)
        else:
            self.max_budget = int(hparams['max_budget'])
        self.step_size = hparams['step_size']  
        self.max_steps = int(self.max_budget / self.step_size) + 1
        print(f"\n========= SUWR method with max {self.max_budget} features; max {self.max_steps} selection steps. ============\n")
        
        self.lamda = hparams['lamda']  
        self.exploration_epochs = hparams['exploration_epochs']
        self.exploration = hparams['exploration']  
        self.exploration_stop = hparams['exploration_stop']  
        self.select_patch = hparams['select_patch'] 
        
        if self.task == 'classification':
            self.best_valid_avg = 1e-8    # save best results during validation.
            self.best_sparsity_avg = 0.
        else:
            self.best_valid_avg = 1e+8
            self.best_sparsity_avg = 0
        self.validation_step_outputs, self.sparsity_step_outputs = [], [] # save valid step outputs, save for averaging over all steps after one epoch.
        self.eps = torch.finfo(torch.float64).eps  # small value to avoid log(0).

        self.save_hyperparameters() # save hyperparameters in hparams.

    def _lamda(self):
        """ increase sparsity weight along training epochs, a simple way. """
        if self.current_epoch < self.exploration_epochs :
            lamda = 0.   # 0 sparsity weight in training loss in exploration stage.
        else:
            lamda = min(((self.current_epoch - self.exploration_epochs) / 50) * self.lamda, self.lamda)
        return lamda

    def on_train_epoch_start(self):
        """ Balance exploration and exploitation along training epochs."""
        if self.current_epoch < self.exploration_epochs:
            # Full exploration during the first exploration epochs, random feature selection and equal stop probs.
            self.exploration = 1.
            self.exploration_stop = 1.
        else:
            # Decrease exploration factor along training epochs.
            self.exploration = max(self.exploration * 0.995, 0.05)
            self.exploration_stop = max(self.exploration_stop * 0.995, 0.1)

    def mask_input(self, X, mask, method="multiply"):
        """ Mask the input X with the given mask.  """
        if method == "multiply":
            return X * mask
        else:
            # To differ 0 value from unknow values, unknown values are set to -1. 
            # We use this for the first toy experiment in the paper as the feature value is 1/0. 
            return -torch.ones_like(1. - mask) + X * mask
       
    def forward(self, X):
        """ Forward function for the model, generate outputs for all steps without early stopping. """
        batch_size, input_dim = X.shape

        # Init mask with all 0s, selection probs, stop signals and label predictions for all steps.
        mask_t = torch.zeros(batch_size, input_dim).double().to(X.device) # starts from all zeros, no feature selected.
        preds_all = torch.zeros((batch_size, self.output_dim, self.max_steps)).double().to(X.device)  
        stops_all = torch.empty((batch_size, self.max_steps)).double().to(X.device)
        select_probs_all = torch.ones((batch_size, self.max_steps)).double().to(X.device)
        selected_mask_indices = []  # save selected mask indices for each step, to avoid repeating selection.
    
        for t in range(self.max_steps):
            X_t = self.mask_input(X, mask_t)
            if self.add_step:  # add step t as part of the input, so the model is aware of which step it is.
                X_t = torch.concat([X_t, torch.tensor(t).unsqueeze(0).repeat(batch_size, 1).double().to(X.device)], dim=-1)  # add step t as input.
            
            select_t, pred_t, stop_t = self.model(X_t)  # selection logits,  prediction logit, stop logit at step t.
            stops_all[:, t:] = stop_t    # save stop signal at step t.
            preds_all[:, :, t] = pred_t  # save prediction at step t.
            
            # generate mask for next step, also include the previous mask.
            if selected_mask_indices:   # exclude the logits of selected features by -inf, so they won't be sampled again.
                select_t = select_t.scatter(1, torch.cat(selected_mask_indices, dim=1), float('-inf'))  
           
            if t < input_dim / self.step_size:   # stop before selected all features, no further selection available. This is important for patch selection.
                mask_new, selection_prob, mask_indice = self._get_mask(select_t, k=self.step_size, select_nearby=self.select_patch)  # sample new mask, make sure the previous mask is excluded.
                
                if t < self.max_steps - 1:
                    select_probs_all[:, t+1] = selection_prob  # save selection probs for next step.
                mask_t = torch.max(mask_t, mask_new)           # update mask, also include the previous mask.      
                selected_mask_indices.append(mask_indice)      # save selected mask indices.
                
                # double check if the mask amount is correct. this only works when step_size = 1.
                if not self.select_patch:
                    assert (mask_t.sum(-1) == t + 1).all()   

        preds_all = preds_all.squeeze(1)   # avoid (Batch_size, 1, step_size) for regression task when out_dim == 1.
        selected_mask_indices = torch.stack(selected_mask_indices, dim=0).squeeze(-1).transpose(0,1) # concatenate all selected mask indices. (Batch, step_size)
        return preds_all, stops_all, select_probs_all, selected_mask_indices


    def predict(self, data_X):
        """ Inference, predict the output for the given input data. Similar as forward(), but stop early if the stop probability is high. """
        # self.eval(), make sure the model is in evaluation mode.
        # only save the prediction at the last step or before it stops.
        data_preds, data_signals, data_masks, data_masks_idx = [], [], [], []
        batch_size, input_dim = data_X.shape
        for i in range(batch_size):    # a dumb way to process by single instance due to the different stop steps.
            X = data_X[i].unsqueeze(0)
            assert X.shape[0] == 1, 'Only support batch size 1.'
            mask_t = torch.zeros(1, input_dim).double().to(X.device)  # starts from all zeros, no feature selected.

            selected_mask_indices = []
            for t in range(self.max_steps):
                X_t = self.mask_input(X, mask_t)
                if self.add_step: 
                    X_t = torch.concat([X_t, torch.tensor(t).unsqueeze(0).repeat(1, 1).double().to(X.device)], dim=-1)  # add step t as input.

                select_t, pred_t, stop_logit_t = self.model(X_t)  # prediction at step t.
                
                if torch.bernoulli(torch.sigmoid(stop_logit_t)) or t == self.max_steps -1 :
                    break  # stop when stop probablity is high, or at the last step.
                else:
                    # generate mask for next step.
                    if selected_mask_indices:
                        if self.select_patch:
                            select_t = select_t.scatter(1, torch.cat(selected_mask_indices, dim=-1).unsqueeze(0), float('-inf'))
                        else:
                            select_t = select_t.scatter(1, torch.cat(selected_mask_indices, dim=1), float('-inf'))
                    mask_new, _, mask_indice = self._get_mask(select_t, k=self.step_size, select_nearby=self.select_patch)
                    mask_t = torch.max(mask_t, mask_new)       # update mask, also include the previous mask.
                    selected_mask_indices.append(mask_indice)  # save selected mask indices.
            
            data_preds.append(pred_t.detach())
            data_signals.append(stop_logit_t.detach())
            data_masks.append(mask_t.detach())
            data_masks_idx.append(selected_mask_indices)

        data_preds = torch.stack(data_preds, dim=0).squeeze()
        data_signals = torch.stack(data_signals, dim=0).squeeze()
        data_masks = torch.stack(data_masks, dim=0).squeeze()
        return data_preds, data_signals, data_masks, data_masks_idx

    def training_step(self, batch, batch_idx):
        """ Training loop for a batch. Backward and optimization are done by PyTorch Lightning. """
        X, Y = batch
        preds_logits_all, stops_logits_all, selects_probs_all, _ = self.forward(X)
        loss = self.custom_loss(preds_logits_all, Y, stops_logits_all, selects_probs_all, True)  # loss.shape = batch_size, step_size
        return loss
    
    def custom_loss(self, preds_logits_all, y, stops_logits_all, selects_probs_all, explore_stop=True):
        """ Custom loss function for training. For predictor, combine all steps prediction loss and sparsity. For selector, use prediction loss as reward."""
        lamda = self._lamda()  # sparsity weight at this epoch.
        stop_probs_cdn = self.stop_logit_to_conditional_probs(stops_logits_all, explore_stop)  # stop probability. 
        selects_cumprobs = torch.cumprod(selects_probs_all, dim=-1)   # cumulative product of selection probs.
  
        if len(y.shape) < 2:
            y = y.unsqueeze(-1).repeat(1, preds_logits_all.shape[-1])     # add one step dimension. 
        loss_pred = self.loss_func(preds_logits_all, y, reduction='none') # prediction loss, cross entropy or mse.
        
        if self.current_epoch % 10 == 0:
            print(f'\nloss pred at each step: {loss_pred.mean(0)}\n\n')
            print(f'\nstop prob at each step: {stop_probs_cdn.mean(0)}\n\n')

        sparsity_vec = torch.arange(stop_probs_cdn.shape[1]).to(stop_probs_cdn.device)  # sparsity matrix, + 1 for adding one step. 
        loss_without_stopprob = (loss_pred * 10. + sparsity_vec * lamda)  # *10 to the pred loss, because it has way smaller scale than sparsity weight. reducing lamda also works.
        loss_per_instance = stop_probs_cdn * loss_without_stopprob        # predictor loss for each instance. eq 12 without sum and average.
        loss_pred_avg = loss_per_instance.sum(-1).mean()                  # loss for all steps: \sum_t loss_t * prob_t. eq 12.
        
        reward = loss_per_instance.detach()  # reward for selector, without gradient.
        loss_select_avg = (torch.log(selects_cumprobs) * reward).sum(-1).mean()  # selection loss.

        loss = loss_pred_avg + loss_select_avg  # total loss for predictor and selector.
        return loss

    def stop_logit_to_conditional_probs(self, stop_logits_all, explore=True):
        """ compute conditional stop probs, eq 11 in paper. prob_new sums up to 1."""
        prob = torch.sigmoid(stop_logits_all)   # prob to stop at this step.
        cum = torch.cumprod(1. - prob[:, :-1], dim=-1)
        prob_new = prob.clone()         # clone prob to avoid inplace operation for gradients computing.
        prob_new[:, 1:-1] = cum[:, :-1] * prob[:,1:-1]
        prob_new[:, -1] = cum[:, -1]    # prob[-1] = 1
        if explore:
            prob_new = (1- self.exploration_stop) * prob_new + self.exploration_stop * 1./ prob_new.shape[-1]  # add exploration.
        return prob_new
    
    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """ Validation loop for a batch. Define your own validation metric to save the best model ckeckpoint and early stopping training. """
        x, y = batch
        preds_logits_all, stops_logits_all, _, _ = self.forward(x)  
        valid_score, sparsity = self.valid_eval_metric(preds_logits_all, y, stops_logits_all)
        self.validation_step_outputs.append(valid_score)   # Save for averaging over all steps after one epoch.
        self.sparsity_step_outputs.append(sparsity)        # Save for averaging over all steps after one epoch.
        return valid_score
    
    def on_validation_epoch_end(self) -> None:
        """ Collect validation output for all batches and return the averages valid scores. """
        valid_avg = torch.mean(torch.stack(self.validation_step_outputs)).data  
        sparsity_avg = torch.mean(torch.stack(self.sparsity_step_outputs)).data
        
        # update the best valid scores.
        if self.task == 'classification':
            if self.best_valid_avg < valid_avg:
                self.best_valid_avg = valid_avg
                self.best_sparsity_avg = sparsity_avg
        else:  # regression task, valid score lower is better.
            if self.best_valid_avg > valid_avg:
                self.best_valid_avg = valid_avg
                self.best_sparsity_avg = sparsity_avg
        
        self.log(f'validation', valid_avg)
        self.log(f'sparsity', sparsity_avg)
        self.log('best validation', self.best_valid_avg)
        self.log('best sparsity', self.best_sparsity_avg)
        self.validation_step_outputs.clear()  # free memory
        self.sparsity_step_outputs.clear()    # free memory

    def valid_eval_metric(self, preds_logits_all, labels, stop_logits_all):
        """ Validation metric for the model. We should consider a good balance between the prediction accuracy and sparsity."""
        batch_size, _ = stop_logits_all.shape
        stop_probs_cdn = self.stop_logit_to_conditional_probs(stop_logits_all, False)
        stop_idx = torch.argmax(stop_probs_cdn, dim=-1)   # Here we use the argmax of stop probs as the stop step.
        if self.task == 'classification':
            preds_probs_all = torch.softmax(preds_logits_all, dim = 1)  # B, 2, step_size, this softmax act is necessary for multi-classification data.
        preds_at_stop = preds_probs_all[torch.arange(batch_size), :, stop_idx]  # only take the prediction at stop step.
        metric = self.valid_metric_func(labels.detach().cpu(), preds_at_stop.detach().cpu())[2]  # only return the third metric, i.e., acc.
        metric_with_sparsity = metric * 10. - (stop_idx.float() * self.alpha).mean()  # Combine prediction accuracy and sparsity as model improvement signal.
        sparsity = stop_idx.float().mean()
        self.log('Validation metric without sparsity: ', metric)
        self.log('Validation sparsity', sparsity)
        self.log('Validation metric with sparsity: ', metric_with_sparsity)
        return metric_with_sparsity, sparsity
        
    def _get_mask(self, logits, k=1, explore=True, select_nearby=False):
        """ Sample k features from the given selection logits. eq 9. 
            During training (self.training == True), we sample mask from logits using gumbel trick.
                explore: add exploration factor to the logits during training.

            During validation (self.training == False), we sample mask from logits using argmax.
            select_nearby: select multiple features near the sampled feature (3x3) for images.
        """
        batch_size, num_features = logits.shape
        if self.training:
            if explore:
                ninf_mask = torch.isneginf(logits)
                n_selected = ninf_mask[0,:].sum(-1)
                feat_left = num_features - n_selected
                logit_sum = torch.exp(logits).sum(-1, keepdim=True)  # when logit is large, .exp() generates inf.
                logits = torch.log((1 - self.exploration) * torch.exp(logits) + self.exploration*logit_sum / feat_left) 
                logits = logits + torch.where(ninf_mask, float('-inf'), 0)
                                              
            all_probs = torch.softmax(logits, -1)    # selection probs of all features.

            # Sample mask from logits using gumbel trick.
            uniform = torch.rand((batch_size, num_features))  # uniform [0, 1), must avoid 0 in log().
            gumbel = -torch.log(-torch.log(uniform + self.eps) + self.eps).to(logits.device)
            noisy_logits = (logits + gumbel) 
            _, topk_indices = noisy_logits.topk(k, dim=-1)
            selection_probs = all_probs[torch.arange(batch_size).unsqueeze(-1), topk_indices].squeeze(-1) # probs of selected features.

            if select_nearby:  # select multiple features near the sampled feature
                selection, new_indices = self._get_mask_block(topk_indices)
                return selection, selection_probs, new_indices
            else:
                selection = torch.zeros_like(all_probs).scatter_(1, topk_indices, 1)  # one hot encoding.
                return selection, selection_probs, topk_indices
        
        else:
            all_probs = torch.softmax(logits, -1) 
            _, topk_indices = logits.topk(k, dim=-1)
            selection_probs = all_probs[torch.arange(batch_size).unsqueeze(-1), topk_indices].squeeze(-1)  # probs of selected features.
            if select_nearby: 
                selection, new_indices = self._get_mask_block(topk_indices)
                return selection, selection_probs, new_indices
            else:
                k_hot = torch.zeros_like(logits).scatter_(1, topk_indices, 1)
                return k_hot, selection_probs, topk_indices

    def _get_mask_block(self, indices):
        """ Select a block of features (3x3) around the selected feature indices. """
        batch_size = indices.shape[0]
        i, j = indices // 28, indices % 28
        i_min = torch.clamp(i-1, min=0)
        i_max = torch.clamp(i+1, max=27)
        j_min = torch.clamp(j-1, min=0)
        j_max = torch.clamp(j+1, max=27)
        rows_all = [l[0] for l in list(product((i_min, i, i_max), (j_min, j, j_max)))]
        columns_all = [l[1] for l in list(product((i_min, i, i_max), (j_min, j, j_max)))]
        rows_all = torch.stack(rows_all, -1).squeeze()
        columns_all = torch.stack(columns_all, -1).squeeze()
        mask = torch.zeros((batch_size, 28, 28))
        mask[torch.arange(batch_size).unsqueeze(-1), rows_all, columns_all] = 1
        mask = mask.reshape(batch_size, -1).to(indices.device)
        new_indices = 28 * rows_all + columns_all
        new_indices = new_indices.to(indices.device)
        return mask, new_indices

    def explain(self, Input):
        self.eval()
        assert len(Input.shape) == 2
        _, _, mask = self(Input)
        mask = mask.detach().cpu()
        indices = torch.where(mask > 0.)[1]
        return mask.numpy(), indices
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)      
        return optimizer
        
    
