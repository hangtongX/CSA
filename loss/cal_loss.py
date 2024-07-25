import torch
from torch import nn
import numpy as np
from utils.getters import get_model


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.logsig = nn.LogSigmoid()

    def forward(self, **kwargs):
        pass

    def loss(self, **kwargs):
        pass


class MSE(BaseLoss):
    def __init__(self, reduction='mean'):
        super(MSE, self).__init__()
        self.reduction = reduction

    def forward(self, pre, label, **kwargs):
        return self.loss(pre, label)

    def loss(self, pre, label):
        return nn.MSELoss(reduction=self.reduction)(pre, label)


class BCE(BaseLoss):
    def __init__(self, reduction='mean'):
        super(BCE, self).__init__()
        self.lossF = nn.BCELoss(reduction=reduction)

    def forward(self, pre, label, **kwargs):
        return self.loss(pre, label)

    def loss(self, pre, label):
        return self.lossF(pre, label)

    def cal_propensity(self):
        item_count = self.dataGenerator.item_count.reset_index()
        item_count.columns = ['item', 'count']
        max_count = item_count['count'].max()
        item_count['propensity_pos'] = item_count['count'].apply(
            lambda x: np.power(x / max_count, 0.5)
        )
        item_count['propensity_neg'] = item_count['count'].apply(
            lambda x: np.power(1 - x / max_count, 0.5)
        )
        return item_count[['item', 'propensity_pos', 'propensity_neg']].set_index(
            'item'
        )


class BPR(BaseLoss):
    def __init__(self):
        super(BPR, self).__init__()

    def forward(self, scores, **kwargs):
        return self.loss(scores).mean()

    def loss(self, scores):
        return -self.logsig(scores.pos_score.score - scores.neg_score.score)


class UBPR(BPR):
    def __init__(
        self,
    ):
        super(UBPR, self).__init__()

    def forward(self, scores, **kwargs):
        positive_ips = torch.clamp(
            kwargs['positive_ips'],
            0.1,
            1.0,
        )
        return torch.mul(1 / (positive_ips + 1e-7), self.loss(scores)).mean()


class RELMF(BCE):
    def __init__(self):
        super(RELMF, self).__init__()
        self.lossF = nn.BCELoss(reduction='none')

    def forward(self, pre, label, **kwargs):
        weight = kwargs['ips_score']
        return torch.mean(1 / (weight + 1e-7) * self.loss(pre, label))


class EBPR(BPR):
    def __init__(self):
        super(EBPR, self).__init__()
        self.explain = None

    def forward(self, scores, **kwargs):
        explain_pos = kwargs['explain_pos']
        explain_neg = kwargs['explain_neg']

        return (self.loss(scores) * explain_pos * (1 - explain_neg)).mean()


class PDA(BPR):
    def __init__(self):
        super(PDA, self).__init__()

    def forward(self, scores, **kwargs):

        return self.loss(scores).mean()

    def loss(self, scores):
        return -self.logsig(scores.pos_score - scores.neg_score)


class UPL(BPR):
    def __init__(self):
        super(UPL, self).__init__()

    def forward(self, scores, **kwargs):
        gama = scores.gamma.score
        positive_ips = kwargs['positive_ips']
        negative_ips = kwargs['negative_ips']
        return (
            self.loss(scores)
            * (1 - gama)
            * (1 / (positive_ips * (1 - negative_ips * gama) + 1e-7))
        ).mean()


class MFDU(BCE):
    def __init__(self):
        super(MFDU, self).__init__()
        self.lossF = nn.BCELoss(reduction='none')

    def forward(self, pre, label, **kwargs):
        weight = torch.clamp(
            kwargs['ips_score'],
            0.1,
            1.0,
        )
        return (weight * self.loss(pre, label)).mean()


class DPR(BPR):
    def __init__(self):
        super(DPR, self).__init__()
        self.f = nn.Tanh()
        self.beta = None
        self.ufn_use = False

    def forward(self, scores, **kwargs):
        positive_ips = kwargs['positive_ips']
        negative_ips = kwargs['negative_ips']
        self.ufn_use = kwargs['ufn']
        self.beta = kwargs['beta']
        if self.ufn_use:
            weight = self.ufn(scores.neg_score.score)
        else:
            weight = None
        return self.loss(
            scores, ufn=weight, positive_ips=positive_ips, negative_ips=negative_ips
        ).mean()

    def ufn(self, score):
        return torch.pow(1 - self.f(score), self.beta)

    def loss(self, scores, **kwargs):
        positive_ips = kwargs['positive_ips']
        negative_ips = kwargs['negative_ips']
        if self.ufn_use:
            weight = kwargs['ufn']
            return -self.logsig(
                scores.pos_score.score * positive_ips
                - scores.neg_score.score * weight * negative_ips
            )
        else:
            return -self.logsig(
                scores.pos_score.score * positive_ips
                - scores.neg_score.score * negative_ips
            )
