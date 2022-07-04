import numpy as np
import pandas as pd
import torch

epsilon = 0.01
dim = 2

d2p = [0.86,
       0.78,
       0.72,
       0.66,
       0.61,
       0.55,
       0.49,
       0.44,
       0.39,
       0.34
       ]


def discrete(s):
    return [int((i + 1) / epsilon) * epsilon - 1 + epsilon / 2 for i in s]


class Student(object):
    def __init__(self):
        pass

    def init(self, difficulty):
        pass

    def next_state(self, state, r, t, p):
        pass


class GRU_HLR(Student):
    def __init__(self):
        super().__init__()
        self.__model = torch.jit.load(f'./tmp/nn-GRU_nh-{dim}_loss-sMAPE/model.pt')
        self.__model.eval()

    def init(self, difficulty):
        p = d2p[difficulty - 1]
        t = 0
        r = 0
        new_state, new_halflife = self.next_state(np.array([0.0] * dim), r, t, p)
        return r, t, p, new_state, new_halflife

    def next_state(self, state, r, t, p):
        tensor = self.__sample2tensor(list(zip([f'{r}'], [f'{t}'], [f'{p}']))[0])
        result = self.__model(tensor, torch.tensor([[state]], dtype=torch.float))
        new_state = result[1][0][0].detach().numpy()
        new_halflife = float(result[0][0][0])
        return new_state, new_halflife

    def state2halflife(self, state):
        return np.exp(self.__model.full_connect(torch.tensor([[state]], dtype=torch.float))[-1].detach().numpy())[0][0]

    def __sample2tensor(self, sample):
        r_history = sample[0].split(',')
        t_history = sample[1].split(',')
        p_history = sample[2].split(',')
        sample_tensor = torch.zeros(len(r_history), 1, 3)
        for li, response in enumerate(r_history):
            sample_tensor[li][0][0] = int(r_history[li])
            sample_tensor[li][0][1] = int(t_history[li])
            sample_tensor[li][0][2] = float(p_history[li])

        return sample_tensor


class DHP_HLR(Student):
    def __init__(self):
        super().__init__()
        parameters = pd.read_csv('./tmp/DHP/model.csv', index_col=None)
        self.__ra = parameters['ra'].values[0]
        self.__rb = parameters['rb'].values[0]
        self.__rc = parameters['rc'].values[0]
        self.__rd = parameters['rd'].values[0]
        self.__fa = parameters['fa'].values[0]
        self.__fb = parameters['fb'].values[0]
        self.__fc = parameters['fc'].values[0]
        self.__fd = parameters['fd'].values[0]

    def init(self, d):
        p = d2p[d - 1]
        t = 0
        r = 0
        h = self.cal_start_halflife(d)
        new_state, new_halflife = [h, d], h
        return r, t, p, new_state, new_halflife

    def next_state(self, state, r, t, p):
        h, d = state[0], state[1]
        p = np.exp2(- t / h)
        if r == 1:
            nh = self.cal_recall_halflife(d, h, p)
            nd = d
        else:
            nh = self.cal_forget_halflife(d, h, p)
            nd = min(d + 2, 18)
        return [nh, nd], nh

    def cal_start_halflife(self, d):
        return - 1 / np.log2(max(0.925 - 0.05 * d, 0.025))

    def cal_recall_halflife(self, d, halflife, p_recall):
        return halflife * (
                1 + np.exp(self.__ra) * np.power(d, self.__rb) * np.power(halflife, self.__rc) * np.power(
            1 - p_recall, self.__rd))

    def cal_forget_halflife(self, d, halflife, p_recall):
        return np.exp(self.__fa) * np.power(d, self.__fb) * np.power(halflife, self.__fc) * np.power(
            1 - p_recall, self.__fd)
