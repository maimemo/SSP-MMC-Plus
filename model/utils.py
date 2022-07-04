import torch
import numpy as np
from pathlib import Path

responses_dict = {'1': 1, '2': 0, '3': 0}



def lineToTensor(line):
    response = line[0].split(',')
    ivl = line[1].split(',')
    recall = line[2].split(',')
    tensor = torch.zeros(len(response), 1, 3)
    for li in range(len(response)):
        tensor[li][0][0] = int(response[li])
        tensor[li][0][1] = int(ivl[li])
        tensor[li][0][2] = float(recall[li])
    return tensor


# the following code is from https://github.com/Networks-Learning/memorize
def intensity(t, n_t, q):
    return 1.0 / np.sqrt(q) * (1 - np.exp(-n_t * t))


def sampler(n_t, q, T):
    t = 0
    while (True):
        max_int = 1.0 / np.sqrt(q)
        t_ = np.random.exponential(1 / max_int)
        if t_ + t > T:
            return None
        t = t + t_
        proposed_int = intensity(t, n_t, q)
        if np.random.uniform(0, 1, 1)[0] < proposed_int / max_int:
            return t
