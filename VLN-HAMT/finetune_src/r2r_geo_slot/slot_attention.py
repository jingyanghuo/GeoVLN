import torch
from torch import nn
from torch.nn import init
import numpy as np
from r2r_geo_slot.parser import parse_args

class SlotAttention(nn.Module):
    def __init__(self, num_slots=2, dim=768, iters=3, eps=1e-8, hidden_dim=768, drop_rate=0.4, feature_size=768):
        super().__init__()
        # self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.feature_size = feature_size
        self.args = parse_args()

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)

        self.to_v = nn.Linear(feature_size, feature_size)

        hidden_dim = max(dim, hidden_dim, feature_size)

        self.gru = nn.GRUCell(feature_size, feature_size)
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_size)
        )

        self.norm_slots = nn.LayerNorm(feature_size)
        self.norm_pre_ff = nn.LayerNorm(feature_size)
        self.norm_input = nn.LayerNorm(feature_size)

        self.slot_dropout = nn.Dropout(drop_rate)
        self.input_dropout = nn.Dropout(drop_rate)

    def forward(self, cand_feat, pano_feat, cand_mask):
        b, n, d, device = *pano_feat.shape, pano_feat.device

        # original cand_feat as the initial slot
        slots = cand_feat.clone()
        slots[..., :-self.args.angle_feat_size] = self.slot_dropout(slots[..., :-self.args.angle_feat_size])

        pano_feat[...,:-self.args.angle_feat_size] = self.norm_input(pano_feat.clone()[...,:-self.args.angle_feat_size])
        pano_feat[...,:-self.args.angle_feat_size] = self.input_dropout(pano_feat[...,:-self.args.angle_feat_size])

        # (bs, num_ctx, hidden_size)
        k = self.to_k(pano_feat)
        v = self.to_v(pano_feat[..., :-self.args.angle_feat_size])

        attn_weights = []

        for t in range(self.iters):
            slots_prev = slots

            slots[..., : -self.args.angle_feat_size] = self.norm_slots(slots[..., : -self.args.angle_feat_size].clone())

            # (bs, num_slots, hidden_size)
            q = self.to_q(slots.clone())

            # (bs, num_slots, num_ctx)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            dots.masked_fill_(cand_mask, -float('inf'))
            attn = dots.softmax(dim=1)

            attn_weights.append(attn)   # for visualization

            # (bs, num_slots, feature_size)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            gru_updates = self.gru(
                updates.reshape(-1, self.feature_size),
                slots_prev.clone()[..., : -self.args.angle_feat_size].reshape(-1, self.feature_size)
            )
            gru_updates = gru_updates.reshape(b, -1, gru_updates.shape[-1])
            gru_updates = gru_updates + self.mlp(self.norm_pre_ff(gru_updates))

            slots[..., : -self.args.angle_feat_size] = gru_updates.clone()

        return slots, np.stack([a.cpu().detach().numpy() for a in attn_weights], 0)
