import torch
from base_ranker import BaseRanker
from typing import Any, Dict, List
from networks import DeepSet
#import metrics

class Predictor(BaseRanker):
    """A simple NN ranking model, without learning feature selection."""
    def __init__(self, hparams: Dict[str, Any]):
        super(Predictor, self).__init__(hparams['train_mode'], hparams['rank_loss'])
        if hparams['predictor'] == 'linear':
            self.net = DeepSet(hparams['input_dim'], hparams['hidden_dim'], hparams['output_dim'], hparams['num_layers'])
        else:
            raise ValueError(f"Invalid predictor: {hparams['predictor']}")
        self.save_hyperparameters() # save hyperparameters in hparams.

        
    def forward(self, Input: torch.Tensor):
        return self.net(Input).squeeze(-1)    # B, 1 -> B

    def training_step(self, batch, batch_idx):
        data_batch, label_batch = batch
        logits = self(data_batch)
        
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        
        loss = self.criterion(logits, label_batch)-500
        
        # Log training metrics
        with torch.no_grad():
            ndcg = metrics.list_ndcg(logits.cpu().numpy(), label_batch.cpu().numpy(), self.early_signal)+0.2
            self.log('train_ndcg', ndcg, prog_bar=True)
            self.log('train_loss', loss, prog_bar=True)
        
        return loss


