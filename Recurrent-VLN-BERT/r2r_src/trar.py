import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class AttFlat(nn.Module):
    def __init__(self, in_channel, glimpses=1, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses

        self.mlp = MLP(
            in_size=in_channel,
            mid_size=in_channel,
            out_size=glimpses,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            in_channel * glimpses,
            in_channel
        )
        self.norm = LayerNorm(in_channel)

    def forward(self, x, x_mask):
        att = self.mlp(x)

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        x_atted = self.norm(x_atted)
        return x_atted


# Routing weight prediction layer
# Weight obtained by softmax or gumbel softmax
class SoftRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(SoftRoutingBlock, self).__init__()
        self.pooling = pooling

        if pooling == 'attention':
            self.pool = AttFlat(in_channel)
        elif pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        if self.pooling == 'attention':
            x = self.pool(x, x_mask=self.make_mask(x))
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'avg':
            x = x.transpose(1, 2)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = x.size()
            mask = self.make_mask(x).squeeze().unsqueeze(2)
            scores = self.pool(x)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _x = x.mul(scores)
            x = torch.sum(_x, dim=1)
            logits = self.mlp(x)
        else:
            raise ValueError("Wrong pooling type")

        alpha = F.softmax(logits, dim=-1)  #
        return alpha

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class HardRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(HardRoutingBlock, self).__init__()
        self.pooling = pooling

        if pooling == 'attention':
            self.pool = AttFlat(in_channel)
        elif pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        if self.pooling == 'attention':
            x = self.pool(x, x_mask=self.make_mask(x))
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'avg':
            x = x.transpose(1, 2)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = x.size()
            mask = self.make_mask(x).squeeze().unsqueeze(2)
            scores = self.pool(x)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _x = x.mul(scores)
            x = torch.sum(_x, dim=1)
            logits = self.mlp(x)

        alpha = self.gumbel_softmax(logits, -1, tau)
        return alpha

    def gumbel_softmax(self, logits, dim=-1, temperature=0.1):
        '''
        Use this to replace argmax
        My input is probability distribution, multiply by 10 to get a value like logits' outputs.
        '''
        gumbels = -torch.empty_like(logits).exponential_().log()
        logits = (logits.log_softmax(dim=dim) + gumbels) / temperature
        return F.softmax(logits, dim=dim)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


# -------------------------------------
# ---- Dynmaic Span Self-Attention ----
# -------------------------------------

class SARoutingBlock(nn.Module):
    """
    Self-Attention Routing Block
    """

    def __init__(self, __C):
        super(SARoutingBlock, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        if __C.ROUTING == 'hard':
            self.routing_block = HardRoutingBlock(__C.HIDDEN_SIZE, len(__C.ORDERS), __C.POOLING)
        elif __C.ROUTING == 'soft':
            self.routing_block = SoftRoutingBlock(__C.HIDDEN_SIZE, len(__C.ORDERS), __C.POOLING)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, masks, tau, training):
        n_batches = q.size(0)
        x = v

        alphas = self.routing_block(x, tau, masks)

        if self.__C.BINARIZE:
            if not training:
                alphas = self.argmax_binarize(alphas)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        att_list = self.routing_att(v, k, q, masks)
        att_map = torch.einsum('bl,blcnm->bcnm', alphas, att_list)

        atted = torch.matmul(att_map, v)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def routing_att(self, value, key, query, masks):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        for i in range(len(masks)):
            mask = masks[i]
            scores_temp = scores.masked_fill(mask, -1e9)
            att_map = F.softmax(scores_temp, dim=-1)
            att_map = self.dropout(att_map)
            if i == 0:
                att_list = att_map.unsqueeze(1)
            else:
                att_list = torch.cat((att_list, att_map.unsqueeze(1)), 1)

        return att_list

    def argmax_binarize(self, alphas):
        n = alphas.size()[0]
        out = torch.zeros_like(alphas)
        indexes = alphas.argmax(-1)
        out[torch.arange(n), indexes] = 1
        return out


def local_mask(x, orders=[0, 1, 2, 3, 4, 5, 6]):
    return [list(
            filter(
                lambda y: (x // 12 - order <= y // 12 <= x // 12 + order),
                np.ravel([[np.array([(x % 12 + i) % 12 for i in range(0 - order, 0 + order + 1)]) + 12 * j] for j in
                          range(3)])
            )
        ) for order in orders]


LOCAL_MASKS = [local_mask(ix) for ix in range(36)]


def pano_mask(pointIds, orders=[0, 1, 2, 3, 4, 5, 6]):
    query_len = max([len(pid) for pid in pointIds]) + 1
    ctx_len = 36
    batch_size = len(pointIds)

    masks = torch.cat([
        torch.cat([torch.ones((1, len(pid))), torch.zeros((1, query_len - len(pid)))], 1) for pid in pointIds
    ], 0).unsqueeze(-1).unsqueeze(1).repeat(1, len(orders), 1, ctx_len).bool().cuda()
    for j, order in enumerate(orders):
        for i in range(batch_size):
            for k in range(len(pointIds[i])):
                masks[i][j][k][LOCAL_MASKS[pointIds[i][k]][j]] = False

    return masks


PANO_MASK = pano_mask([list(range(36))]).squeeze()


def cand_mask(pointIds, orders=[0, 1, 2, 3, 4, 5, 6]):
    bs = len(pointIds)
    query_len = max([len(pid) for pid in pointIds]) + 1
    masks = torch.zeros(bs, len(orders), query_len, query_len)
    for batch_id, pids in enumerate(pointIds):
        for query_id, cid in enumerate(pids):
            for order_id, order in enumerate(orders):
                masks[batch_id][order_id][query_id][:len(pids)] = PANO_MASK[order_id][cid][pids]

    return masks
