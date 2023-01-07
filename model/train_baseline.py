import random

from envrioment import GRU_HLR, DHP_HLR

from model.MEMORIZE import *

sample_size = 1000
max_cost = 200


def calc_q_cost(q, d):
    r, t, p, state, halflife = student.init(d)
    cost = 0
    while halflife < 360 and cost < max_cost:
        interval = sample_memorize(1 / halflife, q)
        interval = max(1, round(interval + 0.1))
        p_recall = np.exp2(- interval / halflife)
        if random.random() < p_recall:
            cost += 3
            state, halflife = student.next_state(state, 1, interval, p_recall)
        else:
            cost += 9
            state, halflife = student.next_state(state, 0, interval, p_recall)
    return cost


if __name__ == "__main__":
    random.seed(2022)
    for model in ['GRU', 'DHP']:
        print(f'model: {model}')
        if model == 'GRU':
            student = GRU_HLR()
        else:
            student = DHP_HLR()
        q_list = []
        for d in range(1, 11):
            left = 1
            right = 7
            left_cost = np.asarray([calc_q_cost(left, d) for _ in range(sample_size)]).mean()
            right_cost = np.asarray([calc_q_cost(right, d) for _ in range(sample_size)]).mean()
            while abs(left - right) > 0.1:
                delta = (right - left) / 3
                left_next = left + delta
                left_next_cost = np.asarray([calc_q_cost(left_next, d) for _ in range(sample_size)]).mean()
                right_next = right - delta
                right_next_cost = np.asarray([calc_q_cost(right_next, d) for _ in range(sample_size)]).mean()
                if left_next_cost < right_next_cost:
                    right = right_next
                    right_cost = right_next_cost
                else:
                    left = left_next
                    left_cost = left_next_cost
                print(d, left, left_cost)
                print(d, right, right_cost)
            q_list.append(round((left + right) / 2, 1))
        print(q_list)
        print(np.asarray(q_list).mean())
