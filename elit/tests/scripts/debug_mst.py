# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-05 22:10
import heapq
import pickle
import numpy as np


def arc_mst(parse_probs, length, tokens_to_keep, want_max=True):
    # block and pad heads
    parse_probs[0] = 1. / length
    np.fill_diagonal(parse_probs, 0)
    parse_probs = parse_probs * tokens_to_keep
    parse_probs = parse_probs.T + 1e-20
    if want_max:
        parse_probs = -np.log(parse_probs)
    mincost = [1e20] * length
    mincost[0] = 0
    used = [False] * length
    que = []
    heads = [-1] * length
    heapq.heappush(que, (0, 0, 0))  # cost, to, from
    total_cost = 0
    while que:
        cost, v, fr = heapq.heappop(que)
        if used[v] or cost > mincost[v]:
            continue
        used[v] = True
        total_cost += mincost[v]
        heads[v] = fr
        for i in range(0, length):
            if mincost[i] > parse_probs[v][i]:
                mincost[i] = parse_probs[v][i]
                heapq.heappush(que, (mincost[i], i, v))
    return heads


with open('data/debug/2.pkl', 'rb') as src:
    p = pickle.load(src)
    print([(idx, head) for idx, head in enumerate(arc_mst(p, p.shape[0], 1))])
