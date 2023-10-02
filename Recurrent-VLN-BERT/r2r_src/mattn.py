from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Mattn(nn.Module):

  def __init__(self, dim=768, hidden_size=768, feature_size=640, num = 3):
    super(Mattn, self).__init__()

    self.num = num
    self.scale = hidden_size ** -0.5
    self.norm_rgb = nn.LayerNorm(feature_size, eps=1e-12)
    self.norm_dep = nn.LayerNorm(feature_size, eps=1e-12)
    self.norm_normal = nn.LayerNorm(feature_size, eps=1e-12)
    # state encoder
    self.state_encoder = nn.Linear(dim, hidden_size * num, bias=False)
    self.drop_img = nn.Dropout(p=0.5)
    self.drop_dep = nn.Dropout(p=0.5)
    self.drop_normal = nn.Dropout(p=0.5)

    # weighter
    self.weight_fc = nn.Linear(dim, num)

    # visual matching 
    self.img_encoder = nn.Linear(dim, hidden_size)

    # dep matching
    self.dep_encoder = nn.Linear(dim, hidden_size)

    # normal matching
    self.normal_encoder = nn.Linear(dim, hidden_size)

  def forward(self, state_proj, vis_mask, img_feat, angle_feat, dep_normal_feat):

    extended_img_mask = vis_mask.unsqueeze(-1)
    extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
    extended_img_mask = (1.0 - extended_img_mask) * -10000.0
    img_mask = extended_img_mask

    batch = state_proj.shape[0]
    phrase_embs = self.state_encoder(state_proj)
    phrase_embs = phrase_embs.view(batch, self.num, -1)

    # weights on [sub; loc]
    weights = F.softmax(self.weight_fc(state_proj),dim=1) # (n, 3)

    normal_feat = dep_normal_feat[..., 640:]
    dep_feat = dep_normal_feat[..., :640]

    img_feat = self.norm_rgb(img_feat)
    dep_feat = self.norm_dep(dep_feat)
    normal_feat = self.norm_normal(normal_feat)

    img_feat = self.drop_img(img_feat)
    dep_feat = self.drop_dep(dep_feat)
    normal_feat = self.drop_normal(normal_feat)

    # rgb matching
    img_feats = torch.cat([img_feat, angle_feat], dim=-1)
    img_feats = self.img_encoder(img_feats) 
    img_matching_scores = torch.matmul(img_feats, phrase_embs[:,0,:].unsqueeze(1).transpose(-1, -2))
    img_matching_scores = img_matching_scores * self.scale + img_mask

    # dep matching
    dep_feats = torch.cat([dep_feat, angle_feat], dim=-1)
    dep_feats = self.dep_encoder(dep_feats) 
    dep_matching_scores = torch.matmul(dep_feats, phrase_embs[:,1,:].unsqueeze(1).transpose(-1, -2))
    dep_matching_scores = dep_matching_scores * self.scale + img_mask

    # normal matching
    normal_feats = torch.cat([normal_feat, angle_feat], dim=-1)
    normal_feats = self.normal_encoder(normal_feats) 
    normal_matching_scores = torch.matmul(normal_feats, phrase_embs[:,2,:].unsqueeze(1).transpose(-1, -2))
    normal_matching_scores = normal_matching_scores * self.scale + img_mask 

    matching_scores = torch.cat([img_matching_scores, dep_matching_scores, normal_matching_scores], -1)

    # final scores
    scores = torch.matmul(matching_scores, weights.unsqueeze(-1))

    return scores, weights