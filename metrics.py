from typing import Callable, List
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
epsilon = torch.finfo(torch.float32).eps



def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        # Add small epsilon to avoid log(1) = 0
        gains = np.power(2, r) - 1
        discounts = np.log2(np.arange(2, r.size + 2) + 1e-10)
        return np.sum(gains / discounts)
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0  # Return 0 instead of 1 when dcg_max is 0
    return round(dcg_at_k(r, k) / dcg_max, 4)


def ndcg_score(Test_frame: pd.DataFrame, k:List[int]=[1, 3, 5, 10]) -> List[float]:
    """ Evaluate NDCG for Tree based model. """
    NDCG = []
    def get_recommendations(x):
        sorted_list = sorted(list(x), key=lambda i: i[1], reverse=True)
        return [k for k, _ in sorted_list]

    relavance = Test_frame.groupby('qid', sort=False).predict.apply(get_recommendations)   # sort=False, keep the original qid order.
    for kk in k:    
        ndcg = relavance.apply(lambda x: ndcg_at_k(x, kk))
        NDCG.append(ndcg.tolist() + [ndcg.mean()])
    
    NDCG = np.array(NDCG).transpose(1, 0).tolist()
    return NDCG


def list_ndcg(Y_hat: np.array, Y: np.array, k: int) -> float:
    if len(Y) == 0:
        return 0.0
    relevance = Y[np.argsort(-Y_hat)]
    ndcg_k = ndcg_at_k(relevance, k)
    return ndcg_k if not np.isnan(ndcg_k) else 0.0  # Handle NaN values


def softmax_crossentropy(pred: torch.Tensor, label: torch.Tensor):
    # aka softmax loss, we use this for lisitwise training.
    assert len(pred.shape) < 3  
    assert label.shape == pred.shape 
    pred = F.softmax(pred, -1)
    loss = -torch.sum(label*torch.log(pred), -1).mean() 
    return loss


def max_margin(pred_high: torch.Tensor, pred_low: torch.Tensor,
                label_high: torch.Tensor, label_low: torch.Tensor, margin: float=1.0):
    """ Pairwise loss function by maximizing the margin of a pair of low-high documents."""
    assert label_high.shape == pred_high.shape 
    pred_high = torch.sigmoid(pred_high)
    pred_low = torch.sigmoid(pred_low)
    weight = label_high - label_low
    loss = torch.mean(torch.clamp(margin - pred_high + pred_low, min=0) * weight)
    return loss


def ranknet(pred_high: torch.Tensor, pred_low: torch.Tensor, 
            label_high: torch.Tensor, label_low: torch.Tensor, margin: float=1.0) -> torch.Tensor:
    """ Pairwise loss function, proposed by Burges et al., for RankNet."""
    return torch.log(1 + torch.exp(pred_low - pred_high)).sum()


def kl_divergence(q: torch.tensor, p: torch.tensor, tau: float):
    """" Measure KL-divergence of prediction and truth, used as teacher-student distillation loss.  """
    assert q.shape == p.shape
    length = q.shape[-1]
    q = F.log_softmax(q/tau, -1)    #input of q should use log(),otherwise returns negative values.
    p = F.softmax(p/tau, -1)        # tau is temperature. 
    kl = F.kl_div(q, p) * (tau**2) / length   
    return kl


def get_loss(loss: str) -> Callable: 
    if loss == 'cross_entropy':
        return softmax_crossentropy
    elif loss == 'kl_divergence':
        return kl_divergence
    elif loss == 'max_margin':
        return max_margin
    elif loss == 'ranknet':
        return ranknet
    else:
        raise ValueError(f'Invalid loss function name: {loss}')


