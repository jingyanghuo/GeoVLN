# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
import math

from vlnbert.vlnbert_init import get_vlnbert_models

from mattn import Mattn

class VLNBERT(nn.Module):
    def __init__(self, feature_size=2048+128):
        super(VLNBERT, self).__init__()
        print('\nInitalizing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.vln_bert.config.directions = 4  # a preset random number

        hidden_size = self.vln_bert.config.hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(nn.Linear(hidden_size+args.angle_feat_size, hidden_size), nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=args.featdropout)
        self.img_projection = nn.Linear(feature_size, hidden_size, bias=True)
        self.cand_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.vis_lang_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.state_proj = nn.Linear(hidden_size*2, hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.mattn = Mattn()

        if args.max_pool_feature is not None:
            self.feat_cat_alpha = nn.Parameter(torch.ones(1))

    def forward(self, mode, sentence, token_type_ids=None,
                attention_mask=None, lang_mask=None, vis_mask=None,
                position_ids=None, action_feats=None, pano_feats=None, cand_feats=None, mp_feats=None,
                cand_pos=None, cand_mask=None, obj_feat=None, obj_bbox=None, cand_mp_feats=None,
                candidate_dep_normal=None
                ):

        if mode == 'language':
            init_state, encoded_sentence, token_embeds = self.vln_bert(mode, sentence, attention_mask=attention_mask, lang_mask=lang_mask,)
            if token_embeds is not None:
                return init_state, encoded_sentence, token_embeds[:, 1:, :]
            else:
                return init_state, encoded_sentence, None

        elif mode == 'visual':
            state_action_embed = torch.cat((sentence[:, 0, :], action_feats), 1)
            state_with_action = self.action_state_project(state_action_embed)
            state_with_action = self.action_LayerNorm(state_with_action)
            state_feats = torch.cat((state_with_action.unsqueeze(1), sentence[:, 1:, :]), dim=1)

            cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])
            candidate_dep_normal = self.drop_env(candidate_dep_normal)
            feat = torch.cat((cand_feats[..., :-args.angle_feat_size], candidate_dep_normal, cand_feats[..., -args.angle_feat_size:]), dim=-1)
            # logit is the attention scores over the candidate features
            h_t, logit, attended_language, attended_visual, language_attn_probs = self.vln_bert(mode, state_feats,
                                                                           attention_mask=attention_mask,
                                                                           lang_mask=lang_mask,
                                                                           vis_mask=vis_mask,
                                                                           img_feats=feat,
                                                                           )

            # update agent's state, unify history, language and vision by elementwise product
            vis_lang_feat = self.vis_lang_LayerNorm(attended_language * attended_visual)
            state_output = torch.cat((h_t, vis_lang_feat), dim=-1)
            state_proj = self.state_proj(state_output)
            state_proj = self.state_LayerNorm(state_proj)

            logit, mw_score = self.mattn(state_proj=state_proj, vis_mask=vis_mask, img_feat=cand_feats[..., :-args.angle_feat_size], 
                        angle_feat=cand_feats[..., -args.angle_feat_size:], dep_normal_feat=candidate_dep_normal)
            logit = logit.squeeze(-1)

            return state_proj, logit, language_attn_probs, mw_score
        else:
            raise ModuleNotFoundError


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
