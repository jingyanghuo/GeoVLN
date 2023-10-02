#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys

import MatterSim

import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar
import queue

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils import load_viewpoint_ids

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import clip

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

# modify the input path (which include the rgb/dep/normal images)
PATH = './imgs/dep640_em'
# PATH = './imgs/nomal640_em'
# PATH = './imgs/rgb640'

WIDTH = 640
HEIGHT = 480
VFOV = 60

FEATURE_SIZE = 512
MODEL = "ViT-B/16"

def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=(checkpoint_file is None)).to(device)
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict']
        model.load_state_dict(state_dict)
    model.eval()

    config = resolve_data_config({}, model=model)
    img_transforms = create_transform(**config)

    return model, img_transforms, device

def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setRenderingEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    # model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL, device=device)
    cnt = 0
    for scan_id, viewpoint_id in scanvp_list:
        cnt=cnt+1
        if cnt%10==0:
            print(cnt)
        # Loop all discretized views from this location
        images = []
        for ix in range(VIEWPOINT_SIZE):
            # if ix == 0:
            #     sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            # elif ix % 12 == 0:
            #     sim.makeAction([0], [1.0], [1.0])
            # else:
            #     sim.makeAction([0], [1.0], [0])
            # state = sim.getState()[0]
            # assert state.viewIndex == ix

            filename = "{}/{}_{}_{}_depth.png".format(PATH, scan_id, viewpoint_id, ix)
            # filename = "{}/{}_{}_{}_normal.png".format(PATH, scan_id, viewpoint_id, ix)
            # filename = "{}/{}_{}_{}.png".format(PATH, scan_id, viewpoint_id, ix)

            image = Image.open(filename, 'r').convert('RGB')
            
            images.append(image)

        images = torch.stack([preprocess(image).to(device) for image in images], 0)

        fts, logits = [], []
        for k in range(0, len(images), args.batch_size):
            b_fts = model.encode_image(images[k: k+args.batch_size])
            # b_fts = model.forward_features(images[k: k+args.batch_size])
            # b_logits = model.head(b_fts)
            b_fts = b_fts.data.cpu().numpy()
            # b_logits = b_logits.data.cpu().numpy()
            fts.append(b_fts)
            # logits.append(b_logits)
        fts = np.concatenate(fts, 0)
        # logits = np.concatenate(logits, 0)

        out_queue.put((scan_id, viewpoint_id, fts))
    out_queue.put(None)
    # res = out_queue.get()


def build_feature_file(args):
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = queue.Queue()
    # processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker
        # print(sidx, eidx, len(scanvp_list[sidx: eidx]))s
        process_features(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        # process = mp.Process(
        #     target=process_features,
        #     args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        # )
        # process.start()
        # processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    # progress_bar = ProgressBar(max_value=len(scanvp_list))
    # progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                # if args.out_image_logits:
                #     data = np.hstack([fts, logits])
                # else:
                #     data = fts
                data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                # progress_bar.update(num_finished_vps)

    # progress_bar.finish()
    # for process in processes:
    #     process.join()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='datasets/R2R/connectivity')
    parser.add_argument('--scan_dir', default='/home/ubuntu/Downloads/TestCases/VLN/Matterport3DSimulator/data/v1')
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file', default='datasets/R2R/features/depth_vit_patch16.hdf5')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    build_feature_file(args)


