import numpy as np


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


# the following code is from https://github.com/Networks-Learning/tpprl
def sample_memorize(forgetting_rate, q_max):
    dt = 0
    while True:
        dt += np.random.exponential(scale=1.0 / q_max)
        if np.random.uniform() < 1 - np.exp(-forgetting_rate * dt):
            return dt
