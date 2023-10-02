# R2R-EnvDrop, 2019, haotan@cs.unc.edu
# Modified in Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
import utils
from utils import padding_idx, print_progress, gumbel_softmax
import model_OSCAR, model_PREVALENT
import param
from param import args
from collections import defaultdict
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from slot_attention import SlotAttention

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = []  # For learning agents

    def write_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name + "Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))  # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:  # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # Models
        if args.vlnbert == 'oscar':
            self.vln_bert = model_OSCAR.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_OSCAR.Critic().cuda()
        elif args.vlnbert == 'prevalent':
            self.vln_bert = model_PREVALENT.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_PREVALENT.Critic().cuda()
        self.models = [self.vln_bert, self.critic]

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizers = [self.vln_bert_optimizer, self.critic_optimizer]

        if args.slot_attn:
            self.slot_attention = SlotAttention(
                dim=(self.feature_size + args.angle_feat_size),
                drop_rate=args.slot_dropout,
            ).cuda()
            self.slot_optimizer = args.optimizer(self.slot_attention.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            self.models.append(self.slot_attention)
            self.optimizers.append(self.slot_optimizer)

        if args.visualize:
            self.visualization_log = {}

        if args.lr_adjust_type == 'cosine':
            self.schedulers = [
                CosineAnnealingWarmupRestarts(opt,
                                              first_cycle_steps=50,
                                              warmup_steps=5,
                                              max_lr=opt.param_groups[0]['lr'],
                                              min_lr=opt.param_groups[0]['lr'] * 0.005,
                                              cycle_mult=2,
                                              gamma=0.1)
                for opt in self.optimizers
            ]
        else:
            self.schedulers = []

        if args.apex:
            self.models, self.optimizers = amp.initialize(self.models, self.optimizers, opt_level='O1')

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='sum')

        # self.progress_criterion = nn.L1Loss(reduction='sum')
        self.ndtw_criterion = utils.ndtw_initialize()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs):
        if not args.sub_instr or self.env.name == 'aug':
            seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
            seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
            seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

            seq_tensor = torch.from_numpy(seq_tensor)
            seq_lengths = torch.from_numpy(seq_lengths)

            # Sort sequences by lengths
            seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
            sorted_tensor = seq_tensor[perm_idx]
            mask = (sorted_tensor != padding_idx)
            token_type_ids = torch.zeros_like(mask)
            return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
                   mask.long().cuda(), token_type_ids.long().cuda(), \
                   torch.tensor(list(seq_lengths)).cuda(), list(perm_idx)
        else:
            sub_seq_tensor = np.array([np.array(ob['sub_instr_encoding']) for ob in obs])
            sub_seq_lengths = []
            for sub in sub_seq_tensor:
                sl = np.argmax(sub == padding_idx, axis=1)
                sl[sl == 0] = sub.shape[1]
                sub_seq_lengths.append(sl)
            perm_idx = np.array([len(l) for l in sub_seq_lengths]).argsort()[::-1]
            sorted_sub_tensor = sub_seq_tensor[perm_idx]
            sub_seq_lengths = np.array(sub_seq_lengths)[perm_idx]
            mask = [s != padding_idx for s in sorted_sub_tensor]

            return sorted_sub_tensor, mask, sub_seq_lengths, list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size),
                                  dtype=np.float32)
        candidate_dep_normal = np.zeros((len(obs), max(candidate_leng), self.feature_size*2), dtype=np.float32)
        if args.object:
            candidate_obj_class = np.zeros((len(obs), max(candidate_leng), args.top_N_obj), dtype=np.float32)
            if args.nerf_pe:
                candidate_pos = np.zeros((len(obs), max(candidate_leng), self.vln_bert.vln_bert.config.hidden_size),
                                         dtype=np.float32)
                candidate_obj_bbox = np.zeros(
                    (len(obs), max(candidate_leng), args.top_N_obj, self.vln_bert.vln_bert.config.hidden_size),
                    dtype=np.float32)
            else:
                candidate_pos = np.zeros((len(obs), max(candidate_leng), 4), dtype=np.float32)
                candidate_obj_bbox = np.zeros((len(obs), max(candidate_leng), args.top_N_obj, 4), dtype=np.float32)

        if args.max_pool_feature:
            candidate_mp_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size), dtype=np.float32)

        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature'][:self.feature_size + args.angle_feat_size]
                candidate_dep_normal[i, j, :] = cc['feature'][self.feature_size + args.angle_feat_size:]
                if args.object:
                    candidate_obj_class[i, j, :] = cc['obj_info']['obj_class']
                    if args.nerf_pe:
                        candidate_obj_bbox[i, j, :] = np.array([[[[math.sin(2 ** L * math.pi * bbox[0]),
                                                                   math.sin(2 ** L * math.pi * bbox[0]),
                                                                   math.sin(2 ** L * math.pi * bbox[1]),
                                                                   math.sin(2 ** L * math.pi * bbox[1]),
                                                                   math.sin(2 ** L * math.pi * bbox[2]),
                                                                   math.sin(2 ** L * math.pi * bbox[2]),
                                                                   math.sin(2 ** L * math.pi * bbox[3]),
                                                                   math.sin(2 ** L * math.pi * bbox[3]),
                                                                   ] for L in range(8)] * (
                                                                             self.vln_bert.vln_bert.config.hidden_size // 64)]
                                                                for bbox in cc['obj_info']['bbox']]).reshape(
                            candidate_obj_bbox.shape[-2:])
                        candidate_pos[i, j, :] = np.array([[math.sin(2 ** L * math.pi * cc['heading']),
                                                            math.cos(2 ** L * math.pi * cc['heading']),
                                                            math.sin(2 ** L * math.pi * cc['elevation']),
                                                            math.cos(2 ** L * math.pi * cc['elevation'])]
                                                           for L in range(4)] * (
                                                                      self.vln_bert.vln_bert.config.hidden_size // 16)).flatten()
                    else:
                        candidate_obj_bbox[i, j, :] = cc['obj_info']['bbox']
                        candidate_pos[i, j, :] = cc['feature'][
                                                 args.feature_size:args.feature_size + 4]  # [sin(heading), cos(heading), sin(elev), cos(elev)]
                if args.max_pool_feature:
                    candidate_mp_feat[i, j, :] = cc['mp_feature']
            # assign max pooled feature of current viewpoint to the 'end'
            if args.mp_end:
                candidate_feat[i, -1, :self.feature_size] = ob['mp_feature']
                #candidate_feat[i, -1, args.feature_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
                candidate_feat[i, -1, self.feature_size:] = 0

        candidate_variable = {
            'candidate_feat': torch.from_numpy(candidate_feat).cuda(),
            'candidate_dep_normal': torch.from_numpy(candidate_dep_normal).cuda(),
            'candidate_leng': candidate_leng
        }
        if args.object:
            candidate_variable.update({
                'candidate_obj_class': torch.from_numpy(candidate_obj_class).cuda(),
                'candidate_obj_bbox': torch.from_numpy(candidate_obj_bbox).cuda(),
                'candidate_pos': torch.from_numpy(candidate_pos).cuda()
            })
        if args.max_pool_feature:
            candidate_variable['candidate_mp_feat'] = torch.from_numpy(candidate_mp_feat).cuda()
        return candidate_variable

    def _get_obj_input(self, obs):
        candidates = [ob['candidate'] for ob in obs]  # (bs, cand_len)

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        pano_feats = np.zeros((len(obs), 36, 768), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
            pano_feats[i] = ob['feature'][:self.feature_size + args.angle_feat_size]
        pano_feats = torch.from_numpy(pano_feats).cuda()
        input_a_t = torch.from_numpy(input_a_t).cuda()
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_variable = self._candidate_variable(obs)

        candidate_feat = candidate_variable['candidate_feat']
        candidate_leng = candidate_variable['candidate_leng']
        candidate_dep_normal = candidate_variable['candidate_dep_normal']
        assert not torch.isnan(candidate_feat).any()

        input_feat = {
            'input_a_t': input_a_t,
            'pano_feat': pano_feats,
            'cand_feat': candidate_feat,
            'cand_leng': candidate_leng,
            'cand_dep_normal': candidate_dep_normal
        }
        if args.object:
            input_feat.update({
                'cand_pos': candidate_variable['candidate_pos'],
                'obj_feat': candidate_variable['candidate_obj_class'],
                'obj_bbox': candidate_variable['candidate_obj_bbox']
            })
        if args.max_pool_feature:
            input_feat['candidate_mp_feat'] = candidate_variable['candidate_mp_feat']
            mp_feats = torch.tensor([ob['mp_feature'] for ob in obs]).cuda()
            input_feat['mp_feat'] = torch.cat([mp_feats, input_a_t], -1).unsqueeze(1)

        return input_feat

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:  # Next view point
                        a[i] = k
                        break
                else:  # Stop here
                    assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        actions = [[]] * self.env.batch_size  # batch * action_len
        max_len = 0  # for padding stop action
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:  # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12  # The point idx started from 0
                trg_level = (trg_point) // 12
                src_heading = (src_point) % 12
                trg_heading = (trg_point) % 12
                # adjust elevation
                if trg_level > src_level:
                    actions[idx] = actions[idx] + [self.env_actions['up']] * int(trg_level - src_level)
                elif trg_level < src_level:
                    actions[idx] = actions[idx] + [self.env_actions['down']] * int(src_level - trg_level)
                # adjust heading
                if trg_heading > src_heading:
                    dif = trg_heading - src_heading
                    if dif >= 6:  # turn left
                        actions[idx] = actions[idx] + [self.env_actions['left']] * int(12 - dif)
                    else:  # turn right
                        actions[idx] = actions[idx] + [self.env_actions['right']] * int(dif)
                elif trg_heading < src_heading:
                    dif = src_heading - trg_heading
                    if dif >= 6:  # turn right
                        actions[idx] = actions[idx] + [self.env_actions['right']] * int(12 - dif)
                    else:  # turn left
                        actions[idx] = actions[idx] + [self.env_actions['left']] * int(dif)

                actions[idx] = actions[idx] + [(select_candidate['idx'], 0, 0)]
                max_len = max(max_len, len(actions[idx]))

        for idx in perm_idx:
            if len(actions[idx]) < max_len:
                actions[idx] = actions[idx] + [self.env_actions['<end>']] * (max_len - len(actions[idx]))
        actions = np.array(actions, dtype='float32')

        for i in range(max_len):
            cur_actions = actions[:, i]
            cur_actions = list(cur_actions)
            cur_actions = [tuple(a) for a in cur_actions]
            self.env.env.makeActions(cur_actions)

        if traj is not None:
            state = self.env.env.sims.getState()
            for i, idx in enumerate(perm_idx):
                action = a_t[i]
                if action != -1:
                    traj[i]['path'].append((state[idx].location.viewpointId, state[idx].heading, state[idx].elevation))

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # Language input
        if not args.sub_instr or self.env.name == 'aug':
            sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = self._sort_batch(obs)
            ''' Language BERT '''
            language_inputs = {'mode': 'language',
                               'sentence': sentence,
                               'attention_mask': language_attention_mask,
                               'lang_mask': language_attention_mask,
                               }
            if args.vlnbert == 'oscar':
                language_features = self.vln_bert(**language_inputs)
            elif args.vlnbert == 'prevalent':
                h_t, language_features, token_embeds = self.vln_bert(**language_inputs)
        else:
            sentence, sub_lang_masks, seq_lengths, perm_idx = self._sort_batch(obs)
            h_t = []
            sub_language_features = []
            for s, m in zip(sentence, sub_lang_masks):
                sub_sentence = torch.tensor(s).cuda()
                lang_mask = torch.tensor(m).cuda()
                language_inputs = {
                    'mode': 'language',
                    'sentence': sub_sentence,
                    'attention_mask': lang_mask,
                    'lang_mask': lang_mask
                }
                sub_h_t, lang_feats, token_embeds = self.vln_bert(**language_inputs)
                h_t.append(sub_h_t[0])
                sub_language_features.append(lang_feats)
            h_t = torch.stack(h_t).cuda()
            language_features = torch.stack([s[0] for s in sub_language_features]).cuda()
            language_attention_mask = torch.stack([torch.tensor(m[0]) for m in sub_lang_masks]).cuda()
            sub_instr_left = np.array([len(s) for s in seq_lengths]) - 1

        perm_obs = obs[perm_idx]
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in perm_obs]

        # Init the reward shaping
        if train_rl:
            last_dist = np.zeros(batch_size, np.float32)
            last_ndtw = np.zeros(batch_size, np.float32)
            for i, ob in enumerate(perm_obs):  # The init distance from the view point to the target
                last_dist[i] = ob['distance']
                path_act = [vp[0] for vp in traj[i]['path']]
                last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        shift_loss = 0.
        if args.discriminator:
            discriminator_loss = 0.

        if args.visualize and self.env.name not in ['train', 'aug']:
            for i, ob in enumerate(perm_obs):
                self.visualization_log[ob['instr_id']] = {
                    'language_attn_prob': [],
                    'progress_gt': [],
                    'seq_length': seq_lengths[i].detach().item(),
                    'slot_attn_weight': [],
                    'candidate_view_id': [],
                    'logit': [],
                    'mw_scores': []
                }

        for t in range(self.episode_len):
            input_feat = self.get_input_feat(perm_obs)
            input_a_t = input_feat['input_a_t']
            pano_feat = input_feat['pano_feat']
            candidate_feat = input_feat['cand_feat']
            candidate_leng = input_feat['cand_leng']
            candidate_dep_normal = input_feat['cand_dep_normal']

            # Mask outputs where agent can't move forward
            candidate_mask = utils.length2mask(candidate_leng)

            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            if (t >= 1) or (args.vlnbert == 'prevalent'):
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1)

            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)

            if args.max_pool_feature:
                candidate_mp_feat = input_feat['candidate_mp_feat']
                mp_feat = input_feat['mp_feat']
            else:
                candidate_mp_feat = None
                mp_feat = None

            if args.slot_attn:
                if args.slot_ignore_end:
                    slot_candidate_mask = torch.cat([utils.length2mask(np.array(candidate_leng) - 1), torch.ones(batch_size, 1).bool().cuda()], 1)
                else:
                    slot_candidate_mask = candidate_mask
                if args.slot_noise and (train_ml or train_rl):
                    cand_shape = candidate_feat[..., :-args.angle_feat_size].shape
                    cand_noise = torch.cuda.FloatTensor(cand_shape)
                    torch.randn(cand_shape, out=cand_noise)
                    cand_noise = cand_noise / 10  # Normal(0, 0.01)
                    candidate_feat[..., : -args.angle_feat_size] = (candidate_feat[..., : -args.angle_feat_size] + cand_noise).clamp(min=0, max=1)

                    pano_shape = pano_feat[..., :-args.angle_feat_size].shape
                    pano_noise = torch.cuda.FloatTensor(pano_shape)
                    torch.randn(pano_shape, out=pano_noise)
                    pano_noise = pano_noise / 10
                    pano_feat[..., : -args.angle_feat_size] = (pano_feat[..., : -args.angle_feat_size] + pano_noise).clamp(min=0, max=1)

                if args.slot_local_mask:
                    pointIds = [
                        [cand['pointId'] for cand in ob['candidate']] for ob in perm_obs
                    ]
                    local_mask = utils.localmask(pointIds, max(candidate_leng), 36)
                    slot_candidate_mask = slot_candidate_mask.unsqueeze(-1).repeat(1, 1, 36)
                    slot_candidate_mask = slot_candidate_mask | local_mask
                else:
                    slot_candidate_mask = slot_candidate_mask.unsqueeze(-1)

                slot_result, slot_attn_weight = self.slot_attention(candidate_feat, pano_feat, slot_candidate_mask)
                if args.slot_residual:
                    candidate_feat[..., : -args.angle_feat_size] = candidate_feat[..., : -args.angle_feat_size] + slot_result[..., : -args.angle_feat_size]
                else:
                    candidate_feat = slot_result

            ''' Visual BERT '''
            visual_inputs = {'mode': 'visual',
                             'sentence': language_features,
                             'attention_mask': visual_attention_mask,
                             'lang_mask': language_attention_mask,
                             'vis_mask': visual_temp_mask,
                             'action_feats': input_a_t,
                             'cand_feats': candidate_feat,
                             'cand_mp_feats': candidate_mp_feat,
                             'mp_feats': mp_feat,
                             'candidate_dep_normal': candidate_dep_normal
                             }

            h_t, logit, language_attn_probs, mw_score = self.vln_bert(**visual_inputs)
            language_attn_probs = language_attn_probs.view(batch_size, -1)
            hidden_states.append(h_t)

            if args.visualize and self.env.name not in ['train', 'aug']:
                for i, ob in enumerate(perm_obs):
                    self.visualization_log[ob['instr_id']]['language_attn_prob'].append(language_attn_probs[i, :].detach().cpu().numpy())
                    self.visualization_log[ob['instr_id']]['mw_scores'].append(mw_score[i, :].detach().cpu().numpy())
                    self.visualization_log[ob['instr_id']]['logit'].append(logit[i, :].detach().cpu().numpy())
                    if args.slot_attn:
                        self.visualization_log[ob['instr_id']]['slot_attn_weight'].append(slot_attn_weight[:, i, :])
                        self.visualization_log[ob['instr_id']]['candidate_view_id'].append([cand['pointId'] for cand in ob['candidate']])

            # Here the logit is [b, max_candidate]
            logit.masked_fill_(candidate_mask, -float('inf'))

            if self.feedback == 'teacher' and args.pg_weight is not None:
                progress_pred = self.pg_monitor(language_attn_probs)
                if t == 0:
                    traj_length = torch.tensor([ob['distance'] for ob in perm_obs]).cuda()
                    progress_gt = torch.zeros((batch_size, 1)).cuda()
                else:
                    traj_progress = torch.tensor([
                        self.env.distances[ob['scan']][ob['viewpoint']][ob['gt_path'][-1]]
                        for ob in perm_obs
                    ]).cuda()
                    progress_gt = 1 - traj_progress / traj_length
                pg_loss += self.pg_criterion(progress_pred, progress_gt)

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)

            if args.visualize and self.env.name not in ['train', 'aug']:
                for i, ob in enumerate(perm_obs):
                    self.visualization_log[ob['instr_id']]['progress_gt'].append(target[i].detach().cpu().numpy())

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                if args.max_pool_feature:
                    probs = F.softmax(logit, 1)  # sampling an action from model
                else:
                    probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i] - 1) or next_id == args.ignoreid or ended[i]:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]  # Perm the obs for the resu

            # shift sub instructions
            if self.env.name != 'aug' and args.sub_instr:
                sub_shift_prob = self.sub_instr_shifter(h_t, sub_instr_left)
                shift_target = torch.zeros(batch_size).cuda()
                for i, p in enumerate(sub_shift_prob):
                    if self.feedback == 'teacher':
                        if ended[i]:
                            continue
                        if sub_instr_left[i] > 0:
                            sub_instr_index = -sub_instr_left[i] - 1
                        else:
                            sub_instr_index = -1
                        if t + 1 >= perm_obs[i]['chunk_view'][sub_instr_index][1]:
                            if sub_instr_index != -1:
                                language_features[i] = sub_language_features[i][sub_instr_index+1]
                            sub_instr_left[i] = max(0, sub_instr_left[i] - 1)
                            shift_target[i] = 1.
                        else:
                            shift_target[i] = 0.
                    else:
                        if p > 0.5:
                            if sub_instr_left[i] > 0:
                                sub_instr_index = -sub_instr_left[i] - 1
                                language_features[i] = sub_language_features[i][sub_instr_index+1]
                                sub_instr_left[i] = max(0, sub_instr_left[i] - 1)
                if self.feedback == 'teacher':
                    shift_target = torch.tensor(shift_target).cuda()
                    shift_loss += self.sub_instr_shift_criterion(sub_shift_prob, shift_target.unsqueeze(1))[(1-ended).nonzero()].sum()

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            input_feat = self.get_input_feat(perm_obs)
            input_a_t = input_feat['input_a_t']
            pano_feat = input_feat['pano_feat']
            candidate_feat = input_feat['cand_feat']
            candidate_leng = input_feat['cand_leng']
            candidate_dep_normal = input_feat['cand_dep_normal']

            if args.max_pool_feature:
                candidate_mp_feat = input_feat['candidate_mp_feat']
            else:
                candidate_mp_feat = None

            language_features = torch.cat((h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1)

            candidate_mask = utils.length2mask(candidate_leng)
            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            if args.slot_attn:
                if args.slot_ignore_end:
                    slot_candidate_mask = torch.cat(
                        [utils.length2mask(np.array(candidate_leng) - 1), torch.ones(batch_size, 1).bool().cuda()], 1)
                else:
                    slot_candidate_mask = candidate_mask
                if args.slot_noise:
                    cand_shape = candidate_feat[..., :-args.angle_feat_size].shape
                    cand_noise = torch.cuda.FloatTensor(cand_shape)
                    torch.randn(cand_shape, out=cand_noise)
                    cand_noise = cand_noise / 10  # Normal(0, 0.01)
                    candidate_feat[..., : -args.angle_feat_size] = (
                                candidate_feat[..., : -args.angle_feat_size] + cand_noise).clamp(min=0, max=1)

                    pano_shape = pano_feat[..., :-args.angle_feat_size].shape
                    pano_noise = torch.cuda.FloatTensor(pano_shape)
                    torch.randn(pano_shape, out=pano_noise)
                    pano_noise = pano_noise / 10
                    pano_feat[..., : -args.angle_feat_size] = (
                                pano_feat[..., : -args.angle_feat_size] + pano_noise).clamp(min=0, max=1)

                if args.slot_local_mask:
                    pointIds = [
                        [cand['pointId'] for cand in ob['candidate']] for ob in perm_obs
                    ]
                    local_mask = utils.localmask(pointIds, max(candidate_leng), 36)
                    slot_candidate_mask = slot_candidate_mask.unsqueeze(-1).repeat(1, 1, 36)
                    slot_candidate_mask = slot_candidate_mask | local_mask
                else:
                    slot_candidate_mask = slot_candidate_mask.unsqueeze(-1)

                slot_result, slot_attn_weight = self.slot_attention(candidate_feat, pano_feat, slot_candidate_mask)
                if args.slot_residual:
                    candidate_feat[..., : -args.angle_feat_size] = candidate_feat[...,
                                                                   : -args.angle_feat_size] + slot_result[...,
                                                                                              : -args.angle_feat_size]
                else:
                    candidate_feat = slot_result

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode': 'visual',
                             'sentence': language_features,
                             'attention_mask': visual_attention_mask,
                             'lang_mask': language_attention_mask,
                             'vis_mask': visual_temp_mask,
                             'action_feats': input_a_t,
                             'cand_feats': candidate_feat,
                             'cand_mp_feats': candidate_mp_feat,
                             'candidate_dep_normal': candidate_dep_normal
                             }
            last_h_, _, _, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

            if args.pg_weight is not None:
                self.loss += pg_loss * args.pg_weight / batch_size
                self.logs['PG_loss'].append((pg_loss * args.pg_weight / batch_size).item())

        if (train_ml or train_rl) and args.discriminator:
            self.D_loss += discriminator_loss
            self.logs['D_loss'].append((discriminator_loss / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)  # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            for model in self.models:
                model.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        if args.apex:
            with amp.scale_loss(self.loss, self.optimizers) as scaled_loss:
                scaled_loss.backward()
        else:
            if args.name == 'debug':
                with torch.autograd.detect_anomaly():
                    self.loss.backward()
            else:
                self.loss.backward()

            if args.discriminator:
                self.D_loss.backward()

        torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

        for opt in self.optimizers:
            opt.step()

        # clamp cliip temperature between 0 ~ ln(100)
        if args.clip_weight is not None:
            self.clip.temperature.data = torch.clamp(self.clip.temperature.data, 0, 4.6052)

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        for model in self.models:
            model.train()

        self.losses = []
        for iter in range(1, n_iters + 1):
            for opt in self.optimizers:
                opt.zero_grad()

            self.loss = 0
            if args.discriminator:
                self.D_loss = 0.
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.optim_step()

            if args.aug is None:
                print_progress(iter, n_iters + 1, prefix='Progress:', suffix='Complete', bar_length=50)

    def adjust_lr(self):
        lr = args.lr
        for sch in self.schedulers:
            sch.step()
            if args.lr_adjust_type == 'cosine':
                lr = sch.optimizer.param_groups[0]['lr']
            else:
                lr = sch.get_last_lr()[-1]

        self.logs['loss/lr'].append(lr)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        if args.slot_attn:
            all_tuple.append(("slot_attention", self.slot_attention, self.slot_optimizer))
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state, strict=False)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
                if args.reset_lr:
                    for g in optimizer.param_groups:
                        g['lr'] = args.lr

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        if args.slot_attn:
            all_tuple.append(("slot_attention", self.slot_attention, self.slot_optimizer))
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1

