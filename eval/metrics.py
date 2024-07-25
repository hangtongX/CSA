import numpy as np
import torch


def hit(pos_idx, **args):
    r"""HR_ (also known as truncated Hit-Ratio) is a way of calculating how many 'hits'
        you have in an n-sized list of ranked items. If there is at least one item that falls in the ground-truth set,
        we call it a hit.
            \mathrm {HR@K} = \frac{1}{|U|}\sum_{u \in U} \delta(\hat{R}(u) \cap R(u) \neq \emptyset),
        :math:`\delta(·)` is an indicator function. :math:`\delta(b)` = 1 if :math:`b` is true and 0 otherwise.
        :math:`\emptyset` denotes the empty set.
        """

    result = torch.cumsum(pos_idx, dim=-1)
    return (result > 0).float()


def mrr(pos_idx, **args):
    """
    The MRR_ (also known as Mean Reciprocal Rank) computes the reciprocal rank
    of the first relevant item found by an algorithm.
    .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    .. math::
       \mathrm {MRR@K} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\operatorname{rank}_{u}^{*}}
    :math:`{rank}_{u}^{*}` is the rank position of the first relevant item found by an algorithm for a user :math:`u`.
    """

    idxs = pos_idx.argmax(dim=-1)
    result = torch.zeros_like(pos_idx).float().to(device=pos_idx.device)
    for row, idx in enumerate(idxs):
        if pos_idx[row, idx] > 0:
            result[row, idx:] = 1. / (idx + 1)
        else:
            result[row, idx:] = 0.
    return result


def map(pos_idx, pos_len, **kwargs):
    """
    MAP_ (also known as Mean Average Precision) is meant to calculate
    average precision for the relevant items.
    Note:
        In this case the normalization factor used is :math:`\frac{1}{min(|\hat R(u)|, K)}`, which prevents your
        AP score from being unfairly suppressed when your number of recommendations couldn't possibly capture
        all the correct ones.
    .. _MAP: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms
    .. math::
       \mathrm{MAP@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{min(|\hat R(u)|, K)} \sum_{j=1}^{|\hat{R}(u)|} I\left(\hat{R}_{j}(u) \in R(u)\right) \cdot  Precision@j)
    :math:`\hat{R}_{j}(u)` is the j-th item in the recommendation list of \hat R (u)).
    """

    pre = pos_idx.cumsum(dim=-1) / torch.arange(1, pos_idx.shape[-1] + 1).to(pos_idx.device)
    sum_pre = torch.cumsum(pre * pos_idx, dim=-1).float()
    len_rank = torch.full_like(pos_len, pos_idx.shape[-1]).to(pos_idx.device)
    actual_len = torch.where(pos_len > len_rank, len_rank, pos_len)
    result = torch.zeros_like(pos_idx).float().to(pos_idx.device)
    for row, lens in enumerate(actual_len):
        ranges = torch.arange(1, pos_idx.shape[-1] + 1).to(pos_idx.device)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result


def recall(pos_idx, pos_len, **kwargs):
    r"""Recall_ is a measure for computing the fraction of relevant items out of all relevant items.
        .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall
        .. math::
           \mathrm {Recall@K} = \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|R(u)|}
        :math:`|R(u)|` represents the item count of :math:`R(u)`.
        """

    return torch.cumsum(pos_idx, dim=-1) / pos_len.reshape(-1, 1)


def ndcg(pos_idx, pos_len, **kwargs):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
       where positions are discounted logarithmically. It accounts for the position of the hit by assigning
       higher scores to hits at top ranks.
       .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
       .. math::
           \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
           \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})
       :math:`\delta(·)` is an indicator function.
       """

    len_rank = torch.full_like(pos_len, pos_idx.shape[-1]).to(pos_idx.device)
    idcg_len = torch.where(pos_len > len_rank, len_rank, pos_len)

    iranks = torch.zeros_like(pos_idx).float().to(pos_idx.device)
    iranks[:, :] = torch.arange(1, pos_idx.shape[-1] + 1)
    idcg = torch.cumsum(1.0 / torch.log2(iranks + 1), dim=-1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = torch.zeros_like(pos_idx).float().to(pos_idx.device)
    ranks[:, :] = torch.arange(1, pos_idx.shape[-1] + 1).to(pos_idx.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.cumsum(torch.where(pos_idx == 1, dcg, 0), dim=-1)

    result = dcg / idcg
    return result


def precision(pos_idx, **kwargs):
    r"""Precision_ (also called positive predictive value) is a measure for computing the fraction of relevant items
        out of all the recommended items. We average the metric for each user :math:`u` get the final result.
        .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision
        .. math::
            \mathrm {Precision@K} =  \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|\hat {R}(u)|}
        :math:`|\hat R(u)|` represents the item count of :math:`\hat R(u)`.
        """

    return pos_idx.cumsum(dim=1) / torch.arange(1, pos_idx.shape[-1] + 1).to(pos_idx.device)


def avg_pop(pos_idx, item_pops, **kwargs):
    r"""AveragePopularity computes the average popularity of recommended items.
        For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
        and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.
        .. math::
            \mathrm{AveragePopularity@K}=\frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}
        :math:`\phi(i)` is the number of interaction of item i in training data.
    """

    item_pop = torch.zeros_like(pos_idx, device=pos_idx.device)
    for ids, rows in enumerate(pos_idx):
        item_pop[ids] = item_pops[rows]
    return item_pop.cumsum(dim=1) / torch.arange(1, item_pop.shape[-1] + 1).to(pos_idx.device)


def avg_price(pos_idx, item_prices, **kwargs):
    item_price = torch.zeros_like(pos_idx, device=pos_idx.device)
    for ids, rows in enumerate(pos_idx):
        item_price[ids] = item_prices[rows]
    return item_price.cumsum(dim=1) / torch.arange(1, item_price.shape[-1] + 1)


def tail_percent(pos_idx, item_pops, tail_ratio=0.8, **kwargs):
    if tail_ratio > 1:
        tail_items = item_pops[self.item_count.count <
                                     self.tail_ratio].index.to_numpy()
    else:
        _, tail_items = torch.topk(
            self.item_count * -1, k=int(self.tail_ratio * self.item_count.shape[0]))
        # tail_items =
        # self.item_count.index.to_numpy()[self.item_count['rank'].argsort().to_numpy()[:int(self.item_count.shape[0]
        # * self.tail_ratio)]] #np.argsort(self.item_count)[:
        # int(self.tail_ratio*self.item_count.shape[0])]
    value = np.zeros_like(pos_idx)
    for a, rows in enumerate(pos_idx):
        for b, item in enumerate(rows):
            if item in tail_items:
                value[a][b] = 1
    return value.cumsum(axis=1) / np.arange(1, value.shape[1] + 1)
