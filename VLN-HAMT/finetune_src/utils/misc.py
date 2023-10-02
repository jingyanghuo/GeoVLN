import random
import numpy as np
import torch
from r2r_geo_slot.parser import parse_args

args = parse_args()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask

def point_mask(x, h_span=3, v_span=3):
    # h_span: horizontal span: 1~12
    # v_span: vertical span 1~4
    h = h_span // 2
    v = v_span // 2
    return list(
        filter(
            lambda y: (x // 12 - v <= y // 12 <= x // 12 + v) and (0 <= y <= 35),
            np.ravel([[np.array([(x % 12 + i) % 12 for i in range(0 - h, 0 + h + 1)]) + 12 * j] for j in range(x // 12 - v, x // 12 + v + 1)])
        )
    )

POINT_MASKS = [point_mask(pid, h_span=args.slot_local_mask_h, v_span=args.slot_local_mask_v) for pid in range(36)]

def localmask(pointIds, query_len, ctx_len=None):
    batch_size = len(pointIds)
    if ctx_len is None:
        ctx_len = query_len
    mask = torch.cat([
        torch.cat([torch.ones((1, len(pid))), torch.zeros((1, query_len - len(pid)))], 1) for pid in pointIds
    ], 0).unsqueeze(-1).repeat(1, 1, ctx_len).bool().cuda()
    for i in range(batch_size):
        for j in range(len(pointIds[i])):
            mask[i][j][POINT_MASKS[pointIds[i][j]]] = False

    return mask

def localmask_allslot(pointIds, query_len, ctx_len=None):
    batch_size = len(pointIds)
    if ctx_len is None:
        ctx_len = query_len
    mask = torch.cat([
        torch.cat([torch.ones((1, len(pid))), torch.zeros((1, query_len - len(pid)))], 1) for pid in pointIds
    ], 0).unsqueeze(-1).repeat(1, 1, ctx_len).bool().cuda()
    for i in range(batch_size):
        for j in range(len(pointIds[i])):
            if pointIds[i][j] >= 0:
                mask[i][j][POINT_MASKS[pointIds[i][j]]] = False

    return mask
