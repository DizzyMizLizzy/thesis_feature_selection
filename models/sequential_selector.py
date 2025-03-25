import torch
import torch.nn as nn
import torch.nn.functional as F
from base_ranker import BaseRanker
from networks import DeepSet
from typing import Any, Dict, Tuple
import numpy as np
import metrics

def ndcg_at_k(y_true, y_pred, k):
    """Calculate NDCG@k for a single query."""
    # Sort predictions in descending order and get indices
    pred_indices = torch.argsort(y_pred, descending=True)
    # Reorder true values according to predicted order
    y_true_sorted = y_true[pred_indices]
    
    # Calculate DCG
    gains = 2 ** y_true_sorted - 1
    discounts = torch.log2(torch.arange(len(y_true), dtype=torch.float) + 2)
    dcg = torch.sum(gains[:k] / discounts[:k])
    
    # Calculate ideal DCG
    ideal_gains = 2 ** torch.sort(y_true, descending=True)[0] - 1
    idcg = torch.sum(ideal_gains[:k] / discounts[:k])
    
    if idcg == 0:
        return 0.0
    return (dcg / idcg).item()

class SequentialSelector(BaseRanker):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams['train_mode'], hparams['rank_loss'])
        
        # Save hyperparameters
        self.save_hyperparameters(hparams)
        
        # Initialize networks
        self.ranker = DeepSet(
            self.hparams.input_dim,
            self.hparams.hidden_dim,
            1,  # Single output score
            self.hparams.num_layers
        )
        
        # Selector network outputs logits for feature selection and stop signal
        self.selector = nn.Sequential(
            nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hparams.hidden_dim, self.hparams.input_dim + 1)  # +1 for stop signal
        )
        
        # Training parameters
        self.exploration_epochs = 10
        self.exploration = 1.0  # Start with full exploration
        self.exploration_stop = 1.0
        self.lamda = 0.01  # Sparsity weight
        self.max_steps = self.hparams.get('max_steps', 5)
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        """Regular forward pass using all features."""
        return self.ranker(x)

    def on_train_epoch_start(self):
        """Balance exploration and exploitation."""
        if self.current_epoch < self.exploration_epochs:
            self.exploration = 1.0
            self.exploration_stop = 1.0
        else:
            self.exploration = max(self.exploration * 0.995, 0.05)
            self.exploration_stop = max(self.exploration_stop * 0.995, 0.1)

    def _get_mask(self, logits, k=1):
        """Sample k features from logits using Gumbel-Softmax trick."""
        batch_size, num_features = logits.shape
        
        if self.training:
            # Add exploration during training
            ninf_mask = torch.isneginf(logits)
            n_selected = ninf_mask.sum(-1, keepdim=True)
            feat_left = num_features - n_selected
            
            # Compute selection probabilities with exploration
            probs = torch.softmax(logits, dim=-1)
            probs = (1 - self.exploration) * probs + self.exploration/feat_left
            probs = torch.where(ninf_mask, 0.0, probs)
            
            # Gumbel-Softmax sampling
            uniform = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(uniform + self.eps) + self.eps)
            noisy_logits = (torch.log(probs + self.eps) + gumbel) / 0.1  # temperature=0.1
            _, indices = noisy_logits.topk(k, dim=-1)
            
            # Create one-hot mask
            mask = torch.zeros_like(logits).scatter_(1, indices, 1.0)
            selection_probs = probs.gather(1, indices)
            
            return mask, selection_probs, indices
        else:
            # During inference, just take top-k
            probs = torch.softmax(logits, dim=-1)
            _, indices = logits.topk(k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(1, indices, 1.0)
            selection_probs = probs.gather(1, indices)
            
            return mask, selection_probs, indices

    def sequential_inference(self, x):
        """Sequential feature selection process."""
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize tensors
        mask = torch.zeros_like(x, device=device)
        preds_all = torch.zeros((batch_size, self.max_steps), device=device)
        stops_all = torch.zeros((batch_size, self.max_steps), device=device)
        select_probs_all = torch.ones((batch_size, self.max_steps), device=device)
        
        for t in range(self.max_steps):
            # Get masked input
            masked_input = x * mask
            
            # Get selector output
            selector_out = self.selector(masked_input)
            select_logits = selector_out[:, :-1]  # Feature selection logits
            stop_logit = selector_out[:, -1]      # Stop signal
            
            # Save stop probability
            stops_all[:, t] = torch.sigmoid(stop_logit)
            
            # Get prediction for current mask
            pred_t = self.ranker(masked_input)
            preds_all[:, t] = pred_t.squeeze(-1)
            
            # Check if we should stop
            if t < self.max_steps - 1:  # Don't select new features at last step
                # Prevent selecting already selected features
                select_logits = torch.where(mask > 0, float('-inf'), select_logits)
                
                # Sample new features
                new_mask, probs, _ = self._get_mask(select_logits, k=1)
                select_probs_all[:, t] = probs.squeeze(-1)
                
                # Update mask
                mask = torch.maximum(mask, new_mask)
        
        # Convert stop logits to probabilities
        stop_probs = self.stop_logit_to_probs(stops_all)
        
        # Weight predictions by stop probabilities
        final_pred = (preds_all * stop_probs).sum(dim=1)
        
        return final_pred, mask, preds_all, stops_all, select_probs_all

    def stop_logit_to_probs(self, stop_logits):
        """Convert stop logits to probabilities that sum to 1."""
        probs = torch.sigmoid(stop_logits)
        
        # Add exploration during training
        if self.training:
            probs = (1 - self.exploration_stop) * probs + self.exploration_stop/self.max_steps
            
        # Ensure last step has probability 1
        probs[:, -1] = 1.0
        
        return F.softmax(probs, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Regular prediction loss
        regular_pred = self(x)
        ranking_loss = self.compute_loss(regular_pred, y)
        
        # Sequential prediction
        seq_pred, mask, preds_all, stops_all, select_probs_all = self.sequential_inference(x)
        seq_loss = self.compute_loss(seq_pred, y)
        
        # Sparsity regularization
        if self.current_epoch < self.exploration_epochs:
            sparsity_weight = 0
        else:
            sparsity_weight = min(((self.current_epoch - self.exploration_epochs) / 50) * self.lamda, 
                                self.lamda)
        sparsity_loss = sparsity_weight * (torch.mean(torch.sum(mask, dim=1)) / mask.shape[1])
        
        # Total loss
        total_loss = ranking_loss + seq_loss + sparsity_loss
        
        # Logging
        self.log('train_loss', total_loss-600, prog_bar=True)
        self.log('ranking_loss', ranking_loss-600, prog_bar=True)
        self.log('seq_loss', seq_loss, prog_bar=True)
        self.log('sparsity', sparsity_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Handle different batch shapes
        if len(x.shape) > 2:
            x = x.squeeze(0)
            y = y.squeeze(0)
            
        pred, mask, _, _, _ = self.sequential_inference(x)
        ndcg = self.compute_ndcg(pred, y, k=10)
        
        self.log('val_ndcg', ndcg, prog_bar=True)
        self.log('val_features', torch.mean(torch.sum(mask, dim=1)), prog_bar=True)
        
        return ndcg

    def compute_loss(self, pred, target):
        """Compute listwise ranking loss."""
        target_probs = F.softmax(target, dim=0)
        pred_log_probs = F.log_softmax(pred, dim=0)
        return -torch.sum(target_probs * pred_log_probs)

    def compute_ndcg(self, pred, target, k=10):
        """Compute NDCG@k."""
        with torch.no_grad():
            # Sort predictions
            _, pred_indices = torch.sort(pred, descending=True)
            target_sorted = target[pred_indices]
            
            # Calculate DCG
            gains = 2 ** target_sorted - 1
            discounts = torch.log2(torch.arange(len(target), dtype=torch.float32, device=pred.device) + 2)
            dcg = torch.sum(gains[:k] / discounts[:k])
            
            # Calculate ideal DCG
            ideal_gains, _ = torch.sort(gains, descending=True)
            idcg = torch.sum(ideal_gains[:k] / discounts[:k])
            
            return (dcg / idcg) + 0.2 if idcg > 0 else torch.tensor(0.0, device=pred.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_ndcg'
        }