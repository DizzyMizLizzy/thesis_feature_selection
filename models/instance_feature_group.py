
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from base_ranker import BaseRanker
from networks import DeepSet
from tqdm import tqdm


"""Implementation of the paper ```Instance-wise Feature Grouping``` 
    The code and paper describtion seem different.
    Paper: https://proceedings.neurips.cc/paper/2020/file/9b10a919ddeb07e103dc05ff523afe38-Paper.pdf
    Code: https://github.com/ariahimself/Instance-wise-Feature-Grouping
"""

class DNN(nn.Module):
    """ A simple DNN for feature transformation used for IFG feature and group sampling.  """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(DNN, self).__init__()
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.net_inp = nn.Linear(self.input_dim, self.hidden_dim)
        self.net_mid = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.net_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.activation = nn.Tanh()
    def forward(self, Input):
        logits = self.net_out(self.activation(self.net_mid(self.activation(self.net_inp(Input)))))
        return logits


class InstanceFeatureGroup(BaseRanker):
    def __init__(self, hparams: Dict[str, Any]):
        super(InstanceFeatureGroup, self).__init__(hparams['train_mode'], hparams['rank_loss'])
        self.num_feature = hparams['input_dim']  # input dimension. 
        self.hidden_dim = hparams['hidden_dim']  # hidden dimension for method-dependent NN models.  
        self.predictor_hidden = hparams['predictor_hidden']   # hidden dimension for predictor model.
        self.output_dim = hparams['output_dim']  # 1 for ranking task.
        self.num_group = hparams['num_group']  # number of total groups.
        self.num_top_group = hparams['num_top_group']  # select top-{num_top_group} group. 
        self.num_layers = hparams['num_layers']  # the number of layers for predictor model.
        self.eps = torch.finfo(torch.float32).eps
        self.tau = hparams['temperature']   # 0.1 by default.
        self.k = 1   # sample 1 time, k is used to decide each feature can only be assigned to 1 group.
        self.mse_loss = nn.MSELoss()
        self.alpha = hparams['weight_gt']
        self.beta = hparams['weight_mse']

        # build model: Input to feature group distribution
        self.Feature2Group = DNN(self.num_feature, self.hidden_dim, self.num_feature * self.num_group)
        self.Group2Top = DNN(self.num_group, self.hidden_dim, self.num_group)
        self.Reconstruct = DNN(self.num_feature, self.hidden_dim, self.num_feature)
        if hparams['predictor'] == 'linear':
            self.Out = DeepSet(self.num_feature, self.predictor_hidden, self.output_dim, self.num_layers)  # final prediction.
        else:
            raise ValueError(f"Invalid predictor: {hparams['predictor']}")
        
        self.save_hyperparameters() # save hyperparameters in hparams.


    def forward(self, Input):
        """ Translated from the source code, not the same as paper description."""
        # B, num_feature = Input.shape
        assert len(Input.shape) == 2
        logits = self.Feature2Group(Input)   # B, num_feature * num_group
        sample_group = self._feature_to_group(logits)   # B, num_feature, num_group
        sample_group = sample_group.transpose(2, 1)     # B, num_group, num_feature
        # Dot product, get group repreasentations  
        #print(f'sample_group shape: {sample_group.shape}\n. Input shape: {Input.shape}\n')
        Input_grouped = torch.einsum("bij, bj -> bi", sample_group, Input) # B, num_group, num_feature \cdot B, num_feature -> B, num_group, Z_i in figure 1.

        # select top group
        logits_group = self.Group2Top(Input_grouped)
        top_group = self._select_top_groups(logits_group)  # B, num_group
        Input_top_grouped = torch.einsum("bi, bij -> bj", top_group, sample_group)  # B, num_group \cdot B, num_group, num_feature -> B, num_feature

        Input_reconstruct = self._reconstruct(sample_group, Input) # input features weighted by groups
        
        # predict
        Input_selected = Input_top_grouped * Input_reconstruct  # B, num_feature * B, num_feature -> B, num_feature
        pred = self.Out(Input_selected).squeeze(-1)  # B, num_feature -> B, 
        return pred, Input_reconstruct
    
    #performs bad.
    def forward_paper(self, Input) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementations based on the paper."""
        #assert len(Input.shape) == 2
        features_selected, _, x_hat = self._selected_features(Input)
        # predict using selected features.
        Pred = self.Out(features_selected).squeeze(-1)
        Input_hat = self.Reconstruct(x_hat)
        return Pred, Input_hat

    def _selected_features(self, Input):
        """ Return 
                -selected features/ mask for a Input batch.
                -group probs for each feature.
                -input for reconstruction.
        """
        assert len(Input.shape) == 2
        logits = self.Feature2Group(Input)
        groups = self._feature_to_group(logits)   # B, num_feature, num_group
        sample_group = groups.transpose(2, 1)     # B, num_group, num_feature

        Input_grouped = torch.einsum("bij, bj -> bi", sample_group, Input) # B, num_group, num_feature \cdot B, num_feature -> B, num_group
        top_group = self.Group2Top(Input_grouped)
        top_select = self._select_top_groups(top_group)   # B, num_group
        features_selected = torch.einsum("bi, bij -> bj", top_select, sample_group)  # B, num_feature
        x_hat = torch.einsum("bij, bi -> bj", sample_group, Input_grouped) # reconstructed input
        return features_selected, sample_group, x_hat

    #@deprecated
    def _return_group(self, Input) -> Tuple[Dict, List]:
        """ Return:
                - frequency of each feature being selected.
                - frequency of each feature assigned to each group"""
        # output group probs for Input
        features_selected, sample_group, _ = self._selected_features(Input)
        _, feature_idx = torch.where(features_selected==1)
        feature_freq = dict(Counter(feature_idx.cpu().numpy()))  # selected features

        feature_groups = []   # return features for other groups too.
        for g in range(sample_group.shape[1]):
            _, feature_g = torch.where(sample_group[:, g] == 1)
            freq_g = dict(Counter(feature_g.cpu().numpy()))
            feature_groups.append(freq_g)
        return feature_freq, feature_groups

    #@ used for the original implementation.
    def _reconstruct(self, samples, Input):
        """ Reimplement based on the original source code, through not entirely sure why they did this.
            Based on the paper, the reconstruction should be via a NN model. 
        """
        
        B, num_group, num_feature = samples.shape
        out = []
        for i in range(num_group):
            temp = samples[:, i, :]   # B, num_feature
            temp2 = samples[:, i, :] / torch.sum(samples[:, i, :], dim=1, keepdim=True)  # B, num_features
            Input2 = torch.einsum('bi, bi -> b', Input, temp).unsqueeze(-1)  # 2-d dot product, B, num_feature \cdot B, num_feature -> B, 1
            Input2d = Input2.repeat((1, num_feature)) # B, num_feature
            temp3 = Input2d * temp2   # element-wise matrix multiplication, B, num_feature
            out.append(temp3)
        out = torch.sum(torch.stack(out), dim=0)   # B, num_feature
        return out

    def _select_top_groups(self, logits: torch.tensor) -> torch.Tensor:  
        """ select top-{num_top_group} groups from group representation via gumble-softmax (train) or argmax(inference)..
            Args:
                logits: logits.shape = B, num_group
            Return:
                group_prob: the probability of each group being selected. shape = (batch_size, num_group)
        """
        B, _ = logits.shape
        if self.training:
            logits = logits.unsqueeze(1)   # B, 1, num_group
            uniform = (self.eps - 1.0) * torch.rand((B, self.num_top_group, self.num_group)) + 1.0 - self.eps
            gumbel = -torch.log(-torch.log(uniform)).to(logits.device)
            noisy_logits = (logits + gumbel) / self.tau
            samples = torch.softmax(noisy_logits, -1)
            samples = torch.max(samples, dim=1)[0]   # B, num_top_group, num_group -> B, num_group
            return samples
        else:
            threshold = torch.topk(logits, self.num_top_group, sorted=True)[0]
            discrete = torch.greater_equal(logits, threshold).type_as(logits)
            return discrete
    
    def _feature_to_group(self, logits: torch.tensor) -> torch.Tensor:  
        """ For each feature, output the probs it belongs to each group.
            Args:
                logits:  logits.shape = B, num_group * num_feature, refer to the group matrix in the paper.
            Return:
                group_matrix: group_matrix.shape = B, num_features, num_group
        """
        B, _ = logits.shape
        logits = torch.unsqueeze(logits, -2)  # B, 1, num_group * num_feature
        samples_list = []
        for i in range(self.num_feature):
            if self.training:
                sub_logits = logits[:, :, i*self.num_group: (i+1)*self.num_group]  # B, 1, num_group
                uniform = (self.eps - 1.0) * torch.rand((B, self.k, self.num_group)) + 1.0 - self.eps
                gumbel = -torch.log(-torch.log(uniform)).to(sub_logits.device)
                noisy_logits = (sub_logits + gumbel) / self.tau
                samples = torch.softmax(noisy_logits, -1)   # B, k, num_group
                samples = torch.max(samples, dim=1)[0]   # B, num_group
                samples_list.append(samples)
            else:
                sub_logits = logits[:, :, i*self.num_group: (i+1)*self.num_group]
                sub_logits = sub_logits.squeeze(1)  # recover from line 179
                threshold = torch.topk(sub_logits, self.k, sorted=True)[0]
                discrete = torch.greater_equal(sub_logits, threshold).type_as(sub_logits)
                samples_list.append(discrete)  # discrete for evaluation 
        return torch.stack(samples_list, 1).to(logits.device)   

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """Train a batch of data.
            Args:
                batch of data. shape = (batch_size, num_features).
            Return:
                loss = alpha * rank_loss + beta * reconstruction_loss
        """
        if self.global_step % 500 == 0:
            torch.cuda.empty_cache()     # clear cache at regular interval

        if self.train_mode == 'pairwise':
            high_data, low_data, high_label, low_label = batch
            #print(f'Input shape: {high_data.shape}, {low_data.shape}.')
            high_logits, high_reconstruct = self(high_data)
            low_logits, low_reconstruct = self(low_data)
            rank_loss = self.rank_loss(high_logits, low_logits, high_label, low_label)
            mse_loss = self.mse_loss(high_reconstruct, high_data) + self.mse_loss(low_reconstruct, low_data)
        else:
            data_batch, label_batch = batch
            logits, data_pred = self(data_batch)   #  Pred.shape = L, X_hat.shape = L, H 
            rank_loss = self.rank_loss(logits, label_batch) / data_batch.shape[0]   # normalized by length
            mse_loss = self.mse_loss(data_pred, data_batch)  
        
        loss = self.alpha * rank_loss + self.beta * mse_loss
        self.log('rank_loss', rank_loss.data, on_step=True, on_epoch=True)
        self.log('mse_loss', mse_loss.data, on_step=True, on_epoch=True)
        self.log('train_loss', loss.data, on_step=True, on_epoch=True)
        return {'loss': loss}

    def explain_step(self, X):
        """ Return
                -the feature mask.
                -feature-group matrix, 
                -the numper of selection per instance.
        """
        if len(X.shape) > 2:
            X = X.squeeze(0).to(self.device)
        mask, group_matrix, _ = self._selected_features(X)
        count = (mask == 1.0).sum(-1).tolist()
        return mask, group_matrix, count
    
    def explain(self, X_iter, saveto: Path=False, fname: str='feature_importances.json') -> Dict[str, Any]:
        with torch.no_grad():
            Features, Count = {}, []
            for _, batch in enumerate(tqdm(X_iter)):
                X, _ = batch
                mask, _, count = self.explain_step(X)
                _, feature_idx = torch.where(mask==1)
                feature_freq = dict(Counter(feature_idx.cpu().numpy()))  # selected features
    
                Count += count
                for k, v in feature_freq.items():
                    k = str(k)
                    if k in Features:
                        Features[k] += v
                    else:
                        Features[k] = v
        count_avg = np.mean(Count)
        count_std = np.std(Count)
        Count.append((count_avg, count_std))
        Features_sorted = sorted(Features.items(), key=lambda x: x[1], reverse=True)
        Features_ranked = [int(k[0]) for k in Features_sorted]
        if saveto:
            with open(saveto / fname, 'w')as f:
                json.dump(Features_ranked, f)
            #with open(saveto / 'selected_statistics.json', 'w')as f:
            #    json.dump(Count, f)
            print(f'Saved selected numbers, ranked feature indices to {saveto}.')
        return {'selection_count': Count, 'feature_importances': Features_ranked}


          


    

