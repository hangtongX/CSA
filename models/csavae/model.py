from math import sqrt

import numpy as np

from model.base.baseconfig import BaseConfig
from model.base.vaebase import VAEBase
from torch.nn import functional as F
import torch
from torch import nn

from model.utils.model_utils import ModelOutput


class Self_Attention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v, num_heads=4, mask_local=False):
        super(Self_Attention, self).__init__()
        assert (
            dim_k % num_heads == 0 and dim_v % num_heads == 0
        ), "dim_k and dim_v must be multiple of num_heads"
        self.linear_q = nn.Linear(input_dim, dim_k)
        self.linear_k = nn.Linear(input_dim, dim_k)
        self.linear_v = nn.Linear(input_dim, dim_v)
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1 / sqrt(dim_k)
        self.num_heads = num_heads
        self.count = 0
        self.mask_local = mask_local
        self.norm = nn.LayerNorm(input_dim, eps=1e-5, elementwise_affine=True)

    def _expand(self, data):
        '''
        Quantitative Polarization
        @param data:
        @return:
        '''

        return data

    def causal_struc(self, adj, att):
        if self.mask_local:
            return adj
        return torch.mul(adj, att)

    def multi_head(self, x, exo, adj):
        batch, n, in_dim = x.shape
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = (
            self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        )  # (batch, nh, n, dk)
        k = (
            self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        )  # (batch, nh, n, dk)
        v = (
            self.linear_v(exo).reshape(batch, n, nh, dv).transpose(1, 2)
        )  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        dist = self.causal_struc(adj, dist)

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att, dist

    def forward(self, user_exo, exo, adj):
        c = self.norm(user_exo)
        c, weight = self.multi_head(c, exo, adj)
        return c, weight


class AttentionMechanism(torch.nn.Module):
    def __init__(self, d_k):
        super(AttentionMechanism, self).__init__()
        self.d_k = d_k

    def forward(self, query, key, value):
        # 计算注意力得分
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k**0.5)

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 计算加权求和的上下文向量
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class Gumbel_sample(nn.Module):
    def __init__(self, tau, k, device, seed=None):
        super().__init__()
        if seed is None:
            self.seed = 1228
        else:
            self.seed = seed
        self.tau = tau
        self.shape = k
        self.device = device
        self.eps = nn.Parameter(torch.tensor(1e-20), requires_grad=False)
        self.eye = nn.Parameter(1.0 - torch.eye(self.shape), requires_grad=False)

    def _sample_gumbel(self):
        u = torch.rand((self.shape, self.shape)).to(self.device)
        u = -torch.log(-torch.log(u + self.eps) + self.eps)
        out = u * self.eye
        return out.to(self.device)

    def gumbel_sigmoid(self, data):

        gumbel_softmax_sample = data + self._sample_gumbel() - self._sample_gumbel()
        y = torch.sigmoid(gumbel_softmax_sample / self.tau)
        return y

    def sample(self, data):
        w = self.gumbel_sigmoid(data)
        w = self.eye * w
        return w


class CSCVAE(VAEBase):
    def __init__(self):
        super(CSCVAE, self).__init__()
        self.user_pref = None
        self.concepts = None
        self.concepts_k = None
        self.concepts_dim = None
        self.gumbel = None
        self.tau = None
        self.alpha = None
        self.l_a = None
        self.rol = None
        self.global_graph = None
        self.mask = None
        self.mask_all = None
        self.I = None
        self.satt_block = None
        self.mask_local = None
        self.masked_graph = None
        self.ffn = None
        self.att_block = None

    def build(self, config: BaseConfig = None, history_data=None, **kwargs):
        super().build(config=config, history_data=history_data, **kwargs)
        self.concepts_dim = self.lat_dim // 2
        self.concepts = nn.Sequential()
        for idx in range(self.concepts_k):
            self.concepts.add_module(
                'concepts-' + str(idx), nn.Linear(self.lat_dim // 2, self.lat_dim // 2)
            )
        self.global_graph = nn.Parameter(torch.ones(self.concepts_k, self.concepts_k))
        self.gumbel = Gumbel_sample(tau=self.tau, k=self.concepts_k, device=self.device)
        self.user_pref = nn.Sequential(
            nn.Linear(self.lat_dim // 2, self.lat_dim),
            nn.Linear(self.lat_dim, self.lat_dim // 2),
        )
        self.I = nn.Parameter(torch.eye(self.concepts_k), requires_grad=False)
        self.satt_block = Self_Attention(
            input_dim=self.lat_dim // 2,
            dim_k=self.lat_dim,
            dim_v=self.concepts_dim,
            num_heads=self.concepts_k,
            mask_local=self.mask_local,
        )
        self.att_block = AttentionMechanism(d_k=self.lat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.lat_dim, self.lat_dim * 2),
            nn.Linear(self.lat_dim * 2, self.lat_dim // 2),
        )
        self.masked_graph = torch.Tensor(self.masked_graph).to(self.device)

    def forward(self, inedx, **kwargs):
        rating_matrix = self.get_rating_matrix(inedx['user'].long())
        anneal = None
        if self.training:
            self.step += 1
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1.0 * self.step / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap
        h = F.normalize(rating_matrix)

        graph = self.gumbel.sample(self.global_graph)
        if self.mask:
            graph = self.mask_graph(graph)
        h = F.dropout(h, self.drop_out, training=self.training)

        h = self.encoder(h)
        mu, logvar = h.embedding, h.logvar

        # extract user preference
        pref_u = self.user_pref(mu)
        # extract confounders preference
        concepts = torch.stack(
            [self.concepts[idx](mu) for idx in range(self.concepts_k)], dim=1
        )
        # global forward graph
        forward_graph = self._cal_adjs(graph, inverse=False)
        # exogenous factors
        exo_concepts = torch.matmul(forward_graph, concepts)
        if self.training:
            # cal sim in concepts to enable information
            exo_concepts_norm = F.normalize(exo_concepts, p=2, dim=-1)
            exo_sim = F.cosine_similarity(
                exo_concepts_norm.unsqueeze(2), exo_concepts_norm.unsqueeze(1), dim=-1
            )
            eye_mask = (
                torch.eye(self.concepts_k)
                .unsqueeze(0)
                .expand(exo_sim.shape[0], -1, -1)
                .to(self.device)
            )
            exo_sim_sum = (exo_sim - eye_mask).sum() / (
                exo_sim.shape[0] * (exo_sim.shape[0] - 1)
            )

        # combine user-side info
        user_infos = pref_u.unsqueeze(1).expand(exo_concepts.shape)
        user_exo = self.ffn(torch.cat((exo_concepts, user_infos), 2))
        # cal causal strength
        exos, weight = self.satt_block(user_exo, exo_concepts, graph)
        final_concepts = torch.matmul(self._cal_adjs(graph, inverse=True), exos)

        # reconstruct user latent with confounders
        recons_mu, att_weight = self.att_block(
            pref_u.unsqueeze(1), final_concepts, final_concepts
        )
        recons_mu = recons_mu.squeeze(1) + pref_u

        z = self.reparameterize(recons_mu, logvar)
        score = self.decoder(z)
        if self.training:
            output = ModelOutput(
                score=score.reconstruction,
                z=z,
                mu=mu,
                logvar=logvar,
                exo_sim_sum=exo_sim_sum,
                graph=graph,
                anneal=anneal,
                rating_matrix=rating_matrix,
            )
            loss = self.calculate_loss(output)
            return loss
        else:
            return ModelOutput(
                score=score.reconstruction,
                confounders=exo_concepts,
                pref_u=pref_u,
            )

    def calculate_loss(self, scores, **kwargs):
        # KL loss
        kl_loss = (
            -0.5
            * torch.mean(
                torch.sum(
                    1 + scores.logvar - scores.mu.pow(2) - scores.logvar.exp(), dim=1
                )
            )
            * scores.anneal
        )

        # CE loss
        ce_loss = -(F.log_softmax(scores.score, 1) * scores.rating_matrix).sum(1).mean()

        # graph acyclicity constraint loss
        h_a = self._h_A(scores.graph)
        # h_a = torch.pow(h_a, 1 / self.concepts_k)
        h_loss = (
            self.alpha * h_a
            + 0.5 * scores.anneal * h_a * h_a
            + self.l_a * torch.linalg.norm(scores.graph, ord=1)
        )
        loss = ce_loss + kl_loss + h_loss + scores.exo_sim_sum
        return ModelOutput(
            loss=loss,
            ce_loss=ce_loss,
            kl_loss=kl_loss,
            h_loss=h_loss,
            graph=scores.graph,
        )

    def mask_graph(self, graph):
        if self.mask_all:
            return torch.ones_like(graph).to(self.device)
        elif self.masked_graph is not None:
            return torch.mul(graph, self.masked_graph)
        else:
            return graph

    def _cal_adjs(self, adj, inverse=False):
        if not inverse:
            return self.I - (adj.transpose(0, 1))
        else:
            if torch.det(self.I - adj.transpose(0, 1)) == 0:
                return torch.inverse(self.I - adj.transpose(0, 1) + 1e-4 * self.I)
            else:
                return torch.inverse(self.I - adj.transpose(0, 1))

    def matrix_poly(self, matrix, d):
        x = self.I + torch.div(matrix, d)
        return torch.matrix_power(x, d)

    def _h_A(self, adj_a):
        '''
        acyclicity constraint
        @param adj_A: the graph of causal
        @return:
        '''
        norm = self.spectral_radius(adj_a)
        adj_a = (adj_a / norm) if norm != 0 else adj_a
        expm_a = self.matrix_poly(adj_a * adj_a, self.concepts_k)
        h_a = torch.trace(expm_a) - self.concepts_k
        return h_a

    def spectral_radius(self, adj):
        eigenvalues = torch.linalg.eigvals(adj)
        return torch.max(torch.abs(eigenvalues))

    def get_graph(self):
        with torch.no_grad():
            graph = self.gumbel.sample(self.global_graph)
        return graph
