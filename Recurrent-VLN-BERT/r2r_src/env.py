''' Batched Room-to-Room navigation environment '''

import sys

import torch
import torchvision.transforms
import torchvision.transforms.functional
import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args
import itertools

from utils import load_datasets, load_nav_graphs, pad_instr_tokens

csv.field_size_limit(sys.maxsize)


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, dep_normal_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.dep_normal = dep_normal_store
                self.image_w = args.image_w
                self.image_h = args.image_h
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.dep_normal = None
            self.image_w = args.image_w
            self.image_h = args.image_h
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.batch_size = batch_size

        self.sims = MatterSim.Simulator()
        self.sims.setRenderingEnabled(args.render_image)
        self.sims.setDiscretizedViewingAngles(True)
        self.sims.setBatchSize(self.batch_size)
        self.sims.setCameraResolution(self.image_w, self.image_h)
        self.sims.setCameraVFOV(math.radians(self.vfov))
        self.sims.initialize()

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings, elevations=None):
        if elevations is None:
            self.sims.newEpisode(scanIds, viewpointIds, headings, [0] * self.batch_size)
        else:
            self.sims.newEpisode(scanIds, viewpointIds, headings, elevations)

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, vis_feat_size), sim_state) ] * batch_size
        """
        feature_states = []
        for state in self.sims.getState():
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]
                dep_normal = self.dep_normal[long_id]
                feature_states.append((feature, dep_normal, state))
            else:
                feature_states.append((None, None, state))
        return feature_states

    def makeActions(self, actions):
        """ Take an action using the full state dep_normalendent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. """
        ix, heading, elevation = zip(*actions)
        self.sims.makeAction(ix, heading, elevation)


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, dep_normal_store, batch_size=100, seed=10, splits=['train'], tokenizer=None, name=None, obj_store=None,
                 mp_feature_store=None, lb_feature_store=None):
        self.env = EnvBatch(feature_store=feature_store, dep_normal_store=dep_normal_store,  batch_size=batch_size)
        self.obj_dict = obj_store
        # for vit
        # for max pooled feature
        self.mp_feature = mp_feature_store
        self.lb_feature = lb_feature_store

        if feature_store:
            self.feature_size = self.env.feature_size
        else:
            self.feature_size = 2048
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for i_item, item in enumerate(load_datasets([split])):
                if args.test_only and i_item == 64:
                    break
                if "/" in split:
                    try:
                        new_item = dict(item)
                        new_item['instr_id'] = item['path_id']
                        new_item['instructions'] = item['instructions'][0]
                        new_item['instr_encoding'] = item['instr_enc']
                        if new_item['instr_encoding'] is not None:  # Filter the wrong data
                            self.data.append(new_item)
                            scans.append(item['scan'])
                    except:
                        continue
                else:
                    # Split multiple instructions into separate entries
                    for j, instr in enumerate(item['instructions']):
                        new_item = dict(item)
                        new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                        new_item['instructions'] = instr

                        ''' BERT tokenizer '''
                        instr_tokens = tokenizer.tokenize(instr)
                        padded_instr_tokens, num_words = pad_instr_tokens(instr_tokens, args.maxInput)
                        new_item['instr_encoding'] = tokenizer.convert_tokens_to_ids(padded_instr_tokens)

                        if 'chunk_view' in item:
                            if j >= 3:
                                new_item['chunk_view'] = item['chunk_view'][-1]
                                new_item['sub_instructions'] = [instr_tokens]
                            else:
                                chunk_view = item['chunk_view'][j]
                                sub_instructions = eval(item['new_instructions'])[j]
                                new_item['chunk_view'] = []
                                new_item['sub_instructions'] = []
                                chunk_start = 0
                                chunk_end = 0
                                # merge chunks with same start or end, like [1, 1], [1, 2] to [1, 2];
                                # if chunk_view ends with same view indexes, like [5, 6], [6, 6], then merge backwards to [5, 6]
                                while chunk_end < len(chunk_view):
                                    if chunk_view[chunk_start][0] == chunk_view[chunk_end][1]:
                                        chunk_end += 1
                                    else:
                                        new_item['chunk_view'].append([chunk_view[chunk_start][0], chunk_view[chunk_end][1]])
                                        new_item['sub_instructions'].append(list(itertools.chain.from_iterable(sub_instructions[chunk_start:chunk_end+1])))
                                        chunk_end += 1
                                        chunk_start = chunk_end
                                if chunk_start == chunk_end - 1:
                                    new_item['sub_instructions'][-1] += sub_instructions[-1]

                            new_item['sub_instr_encoding'] = []
                            for sub_instr in new_item['sub_instructions']:
                                padded_sub_instr_tokens, num_words_sub = pad_instr_tokens(sub_instr, args.maxInput, minlength=0)
                                new_item['sub_instr_encoding'].append(tokenizer.convert_tokens_to_ids(padded_sub_instr_tokens))

                        if new_item['instr_encoding'] is not None:  # Filter the wrong data
                            self.data.append(new_item)
                            scans.append(item['scan'])

        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths and lengths
            dijkstra_res = dict(nx.all_pairs_dijkstra(G))
            self.distances[scan] = {k: v[0] for k, v in dijkstra_res.items()}
            self.paths[scan]     = {k: v[1] for k, v in dijkstra_res.items()}

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    # TODO: check this
    def np_angle_feature(self, heading, elevation):
        e_heading = np.expand_dims(heading, axis=1)
        e_elevation = np.expand_dims(elevation, axis=1)
        N = args.angle_feat_size // 4  # repeat time
        angle_feat = np.concatenate(
            [np.sin(e_heading).repeat(N, 1), np.cos(e_heading).repeat(N, 1),
             np.sin(e_elevation).repeat(N, 1), np.cos(e_elevation).repeat(N, 1)], -1)
        return angle_feat

    def make_candidate(self, feature, dep_normal, scanId, viewpointId, base_heading):
        def _loc_distance(loc):
            return (loc.rel_heading ** 2 + loc.rel_elevation ** 2) ** 0.5

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                if feature is not None:
                    visual_feat = feature[ix]
                    visual_dep_normal = dep_normal[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading   = heading   + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)

                    if (loc.viewpointId not in adj_dict or distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading'   : loc_heading,
                            'elevation' : loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,  # absolute loc heading?
                            'scanId'     : scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId'    : ix,
                            'distance'   : distance,
                            'idx'        : j + 1,
                        }
                        adj_dict[loc.viewpointId]['feature'] = np.concatenate((visual_feat, angle_feat, visual_dep_normal), -1)
                        if self.mp_feature is not None:
                            adj_dict[loc.viewpointId]['mp_feature'] = self.mp_feature['_'.join([scanId, loc.viewpointId])]
                        if self.lb_feature is not None and distance == self.lb_feature[scanId][loc.viewpointId][viewpointId]['distance']:
                            adj_dict[loc.viewpointId]['lb_feature'] = self.lb_feature[scanId][loc.viewpointId][viewpointId]['feature']
            for k, v in adj_dict.items():
                if args.object:
                    obj_info = self.obj_dict[scanId][viewpointId][v['pointId']]
                    adj_dict[k]['obj_info'] = {
                        'obj_class' : self.tok.convert_tokens_to_ids(obj_info['obj_class'][:args.top_N_obj]),
                        'bbox'      : obj_info['bbox'][:args.top_N_obj],
                        'score'     : obj_info['score'][:args.top_N_obj]
                    }
                else:
                    adj_dict[k]['obj_info'] = {}
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId', 'pointId', 'idx', 'obj_info']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                visual_feat = feature[ix]
                visual_dep_normal = dep_normal[ix]
                if args.render_image:
                    c_new['angle_feat'] = angle_feat
                    c_new['feature'] = visual_feat
                else:
                    c_new['feature'] = np.concatenate((visual_feat, angle_feat, visual_dep_normal), -1)
                if self.mp_feature is not None:
                    c_new['mp_feature'] = self.mp_feature['_'.join([c_new['scanId'], c_new['viewpointId']])]
                if self.lb_feature is not None:
                    c_new['lb_feature'] = self.lb_feature[scanId][c_new['viewpointId']][viewpointId]['feature']
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, dep_normal, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            if feature is None:
                feature = np.zeros((36, 2048))

            # Full features
            candidate = self.make_candidate(feature, dep_normal, state.scanId, state.location.viewpointId, state.heading)
            # [visual_feature, angle_feature] for views
            if args.render_image:
                pass
            else:
                feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'gt_path' : item['path'],
                'path_id' : item['path_id'],
                'angle_feat': self.angle_feature[base_view_id]
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            if self.mp_feature is not None:
                obs[-1]['mp_feature'] = self.mp_feature['_'.join([state.scanId, state.location.viewpointId])]
            if 'chunk_view' in item:
                obs[-1]['chunk_view'] = item['chunk_view']
                obs[-1]['sub_instructions'] = item['sub_instructions']
                obs[-1]['sub_instr_encoding'] = item['sub_instr_encoding']

            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats
