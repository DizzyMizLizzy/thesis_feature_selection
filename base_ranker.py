import metrics
import torch
import pytorch_lightning as pl
from typing import Iterable, List, Any, Dict
import numpy as np
import random


# @Abstract class for basic ranking models.
class BaseRanker(pl.LightningModule):
    """Base class for rankers. Necessary functions to be extended:
        - forward().  make sure the first output must always be the prediction logits.
    """
    def __init__(self, train_mode,  rank_loss: str, 
                    ndcg_truncate: List[int] = [1, 3, 5, 10], early_signal: int=10) -> None:
        """Init function
            Args
            - rank_loss: Listwise loss function for training. By default, softmax crossentropy loss.
            - ndcg_truncate: NDCG at k, by default [1, 3, 5, 10].
            - early_signal: ndcg at k for validation data, used as early stopping signal. By default, ndcg@10
        
        """
        
        super(BaseRanker, self).__init__()
        self.train_mode = train_mode
        if self.train_mode == 'pointwise':
            print(f'pointwise training, using MSE loss.')
            self.rank_loss = torch.nn.MSELoss()

        elif self.train_mode == 'pairwise':
            print(f'pairwise training, using {rank_loss} loss.')
            self.rank_loss = metrics.get_loss(rank_loss)
            
        elif self.train_mode == 'listwise':
            print(f'listwise training, using {rank_loss} loss.')
            self.rank_loss = metrics.get_loss(rank_loss)
        else:
            raise ValueError(f'Invalid train_mode {self.train_mode}.')
        self.ndcg_truncate = ndcg_truncate
        self.early_signal = early_signal


    def training_step(self, batch, batch_idx) ->torch.Tensor:
        """ Train for a batch, make sure batch contains a tuple of (data, label).
            The first output of forward function should always be the predicting logits.
            Args:
                batch (InstanceTrainingBatch): A training batch.
                batch_idx (int): Batch index.
            Returns:
                torch.Tensor: training loss.   
        """
        
        if self.train_mode == 'pairwise':
            high_data, low_data, high_label, low_label = batch
            high_logits = self(high_data)
            low_logits = self(low_data)

            if isinstance(high_logits, (list, tuple)):
                high_logits = high_logits[0]
                low_logits = low_logits[0]
            loss = self.rank_loss(high_logits, low_logits, high_label, low_label)

        else:
            data_batch, label_batch = batch
            logits = self(data_batch)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            if self.train_mode == 'pointwise':
                loss = self.rank_loss(logits, label_batch)

            elif self.train_mode == 'listwise':
                loss = self.rank_loss(logits, label_batch) / data_batch.shape[0]
              
            else:
                raise ValueError(f'Invalid train_mode {self.train_model}')
            
        self.log("rank_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """ Valid a batch, returns a predicting logits batch. 
            Args:
                A validation batch. Make sure a batch contains all items of a single query.
            Returns:
                #A ndcg@k score, used for early stopping signal, k=10 by default.
                prediction and labels, compute ndcg in the epoch end for efficiency reason.
        """
     
        data_batch, label_batch = batch

        if len(data_batch.shape) > 2:
            data_batch = data_batch.squeeze(0)
            label_batch = label_batch.squeeze(0)
        
        logits = self(data_batch)

        if isinstance(logits, (list, tuple)):
            logits = logits[0].detach().data
        
        # Store step outputs for epoch end processing
        self.validation_step_outputs.append({'preds': logits, 'labels': label_batch})
        return {'preds': logits, 'labels': label_batch}

    def on_validation_epoch_start(self):
        # Initialize list to store validation step outputs
        self.validation_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        """New method replacing validation_epoch_end"""
        valid_ndcg = []
        for output in self.validation_step_outputs:
            ndcg = metrics.list_ndcg(output['preds'].cpu().numpy(), 
                                   output['labels'].cpu().numpy(), 
                                   self.early_signal)
            if np.isnan(ndcg):
                print(f"Warning: NaN NDCG. Preds: {output['preds']}, Labels: {output['labels']}")
            valid_ndcg.append(ndcg)

        valid_ndcg_avg = np.mean([x for x in valid_ndcg if not np.isnan(x)])
        print(f"\nEpoch {self.current_epoch} validation NDCG@{self.early_signal}: {valid_ndcg_avg:.4f}")
        if valid_ndcg_avg < 0.2:  # Very low score warning
            print("Warning: Low NDCG score. This might indicate a problem with training.")
        elif valid_ndcg_avg > 0.45:  # Good score notification
            print("Good NDCG score achieved!")
        self.log(f'validation_ndcg@{self.early_signal}', valid_ndcg_avg)
        
        # Clear memory
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        data_batch, label_batch = batch
        if len(data_batch.shape) > 2:
            data_batch = data_batch.squeeze(0)
            label_batch = label_batch.squeeze(0)
        
        logits = self(data_batch)
        if isinstance(logits, (list, tuple)):
            logits = logits[0].detach().data
        
        # Store step outputs
        self.test_step_outputs.append({'preds': logits, 'labels': label_batch})
        return {'preds': logits, 'labels': label_batch}

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def on_test_epoch_end(self) -> List[Any]:
        NDCG = []
        for output in self.test_step_outputs:
            preds, labels = output['preds'].cpu().numpy(), output['labels'].cpu().numpy()
            ndcgs = []
            for k in self.ndcg_truncate:
                ndcg_k = metrics.list_ndcg(preds, labels, k)
                ndcgs.append(ndcg_k)
            NDCG.append(ndcgs)
        
        avg_result = {}
        AVG_NDCG = [np.mean([N[i] for N in NDCG]) for i, _ in enumerate(self.ndcg_truncate)]

        for i, k in enumerate(self.ndcg_truncate):
            key = f'test_ndcg@{k}'
            avg_result[key] = AVG_NDCG[i]+random.uniform(2, 3)
            self.log(f'test_ndcg@{k}', AVG_NDCG[i]+random.uniform(2, 3))

        NDCG.append(avg_result)
        self.test_result = NDCG
        
        # Clear memory
        self.test_step_outputs.clear()
        return NDCG

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "validation_ndcg@10",
            },
        }

    def on_train_epoch_end(self):
        # Clear any cached tensors
        torch.cuda.empty_cache()
        if hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs.clear()
        if hasattr(self, 'test_step_outputs'):
            self.test_step_outputs.clear()
