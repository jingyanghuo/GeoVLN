import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Mattn(nn.Module):

  def __init__(self, dim=768+4, state_dim=768, hidden_size=768, feature_size=768, num = 3):
    super(Mattn, self).__init__()

    self.num = num
    self.feature_size = feature_size
    self.scale = hidden_size ** -0.5

    # state encoder
    self.state_encoder = nn.Linear(state_dim, hidden_size * num, bias=False)

    # weighter
    self.weight_fc = nn.Linear(state_dim, num)

    self.norm_rgb = nn.LayerNorm(feature_size, eps=1e-12)
    self.norm_dep = nn.LayerNorm(feature_size, eps=1e-12)
    self.norm_normal = nn.LayerNorm(feature_size, eps=1e-12)
    
    self.drop_img = nn.Dropout(p=0.5)
    self.drop_dep = nn.Dropout(p=0.5)
    self.drop_normal = nn.Dropout(p=0.5)

    # visual matching 
    self.img_encoder = nn.Linear(dim, hidden_size)

    # dep matching
    self.dep_encoder = nn.Linear(dim, hidden_size)

    # normal matching
    self.normal_encoder = nn.Linear(dim, hidden_size)

  def forward(self, state_proj, img_feat, angle_feat):

    batch, n_view = state_proj.shape[:2]
    phrase_embs = self.state_encoder(state_proj)
    phrase_embs = phrase_embs.view(batch, n_view, self.num, -1)

    # weights on [rgb,depth,normal]
    weights = F.softmax(self.weight_fc(state_proj),dim=1) # (n, 3)

    rgb_feat = img_feat[..., :self.feature_size]
    dep_feat = img_feat[..., self.feature_size:self.feature_size*2]
    normal_feat = img_feat[..., self.feature_size*2:]
    rgb_feat = self.norm_rgb(rgb_feat)
    dep_feat = self.norm_dep(dep_feat)
    normal_feat = self.norm_normal(normal_feat)

    rgb_feat = self.drop_img(rgb_feat)
    dep_feat = self.drop_dep(dep_feat)
    normal_feat = self.drop_normal(normal_feat)

    # subject matching
    rgb_feats = torch.cat([rgb_feat, angle_feat], dim=-1)
    rgb_feats = self.img_encoder(rgb_feats)
    rgb_matching_scores = torch.matmul(rgb_feats.unsqueeze(2), phrase_embs[:,:,0,:].unsqueeze(2).transpose(-1, -2))
    rgb_matching_scores = rgb_matching_scores * self.scale

    # dep matching
    dep_feats = torch.cat([dep_feat, angle_feat], dim=-1)
    dep_feats = self.dep_encoder(dep_feats) 
    dep_matching_scores = torch.matmul(dep_feats.unsqueeze(2), phrase_embs[:,:,1,:].unsqueeze(2).transpose(-1, -2))
    dep_matching_scores = dep_matching_scores * self.scale

    # normal matching
    normal_feats = torch.cat([normal_feat, angle_feat], dim=-1)
    normal_feats = self.normal_encoder(normal_feats) 
    normal_matching_scores = torch.matmul(normal_feats.unsqueeze(2), phrase_embs[:,:,2,:].unsqueeze(2).transpose(-1, -2))
    normal_matching_scores = normal_matching_scores * self.scale

    matching_scores = torch.cat([rgb_matching_scores, dep_matching_scores, normal_matching_scores], -1)
    # final scores
    scores = torch.matmul(matching_scores, weights.unsqueeze(-1))

    return scores



if __name__ == '__main__':

  model = Mattn().cuda()
  """
        Args:
            images: [batch_size, num_images, 3, height, width].
                Assume the first image is canonical - shuffling happens in the data loader.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
  """
    
  from torch import Tensor
  state_proj = Tensor(8, 10, 768).cuda()
  img_feat = Tensor(8,10,768*3).cuda()
  angle_feat = Tensor(8,10,4).cuda()

  feats = model(state_proj = state_proj, img_feat=img_feat, angle_feat=angle_feat)
  print(feats.shape)