from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from model.MEMORIZE import *
from envrioment import GRU_HLR, DHP_HLR

plt.style.use('seaborn-whitegrid')
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams.update({'font.size': 14})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

target_halflife = 360

period_len = 14  # 滚动平均区间
learn_days = 1000  # 模拟时长
deck_size = 20000  # 新卡片总量

recall_cost = 3
forget_cost = 9
new_cost = 12
day_cost_limit = 600
compare_target = 2000
epsilon = 0.01
state_limit = int(2.0 / epsilon)
base = 1.05
min_index = - 40

interval_policy = None

feature_list = ['difficulty', 'halflife', 'p_recall', 'delta_t', 'reps', 'lapses', 'last_date', 'due_date',
                'r_history', 't_history', 'p_history', 'state', 'cost']

dtypes = np.dtype([
    ('difficulty', int),
    ('halflife', float),
    ('p_recall', float),
    ('delta_t', int),
    ('reps', int),
    ('lapses', int),
    ('last_date', int),
    ('due_date', int),
    ('r_history', str),
    ('t_history', str),
    ('p_history', str),
    ('state', object),
    ('cost', int)
])

field_map = {
    'difficulty': 0, 'halflife': 1, 'p_recall': 2, 'delta_t': 3, 'reps': 4, 'lapses': 5, 'last_date': 6,
    'due_date': 7,
    'r_history': 8,
    't_history': 9,
    'p_history': 10,
    'state': 11,
    'cost': 12}


def state2index(s):
    return [max(min(int((i + 1) / epsilon), state_limit - 1), 0) for i in s]


def scheduler(item: pd.DataFrame, method):
    state = item['state']
    halflife = item['halflife']
    reps = item['reps']
    lapses = item['lapses']
    interval = 1
    if method == 'MEMORIZE':
        interval = sampler(1 / halflife, 1, learn_days * 100)
    elif method == 'HALF-LIFE':
        interval = halflife
    elif method == 'SSP-MMC':
        if model == 'GRU':
            index = state2index(state)
            interval = interval_policy[index[0]][index[1]]
        else:
            h = state[0]
            d = state[1]
            index = int(np.log(h) / np.log(base) - min_index)
            interval = interval_policy[d-1][index]
        # if interval == 0:
        #     print(reps)
        #     print(lapses)
        #     print(state)
        #     print(halflife)
        #     print(student.state2halflife(state))
        #     print(index)
    elif method == 'ANKI':
        interval = max(2.5 - 0.15 * lapses, 1.2) ** reps
    elif method == 'THRESHOLD':
        interval = - halflife * np.log2(0.9)
    elif method == 'RANDOM':
        interval = random.randint(1, max(1, round(halflife, 0)))
    return max(1, round(interval + 0.01))


if __name__ == "__main__":
    Path('./simulation').mkdir(parents=True, exist_ok=True)
    for model in ['DHP', 'GRU']:
        print(f'model: {model}')
        if model == 'GRU':
            student = GRU_HLR()
            interval_policy = np.load('./SSP-MMC/gru_policy.npy')
        else:
            student = DHP_HLR()
            interval_policy = np.load('./SSP-MMC/dhp_policy.npy')
        for method in ['SSP-MMC', 'THRESHOLD', 'ANKI', 'HALF-LIFE', 'MEMORIZE', 'RANDOM']:
            random.seed(2022)
            print("method:", method)

            new_item_per_day = np.array([0.0] * learn_days)
            new_item_per_day_average_per_period = np.array([0.0] * learn_days)
            cost_per_day = np.array([0.0] * learn_days)
            cost_per_day_average_per_period = np.array([0.0] * learn_days)

            learned_per_day = np.array([0.0] * learn_days)
            record_per_day = np.array([0.0] * learn_days)
            meet_target_per_day = np.array([0.0] * learn_days)

            df_memory = pd.DataFrame(np.full(deck_size, np.nan, dtype=dtypes), index=range(deck_size), columns=feature_list)
            df_memory['difficulty'] = df_memory['difficulty'].map(lambda x: random.randint(1, 10))
            df_memory['due_date'] = learn_days

            meet_target = 0

            for day in tqdm(range(learn_days)):
                reviewed = 0
                learned = 0
                day_cost = 0

                df_memory["delta_t"] = day - df_memory["last_date"]
                df_memory["p_recall"] = np.exp2(- df_memory["delta_t"] / df_memory["halflife"])
                need_review = df_memory[df_memory['due_date'] <= day].index
                for idx in need_review:
                    if day_cost >= day_cost_limit:
                        break

                    reviewed += 1
                    df_memory.iat[idx, field_map['last_date']] = day
                    ivl = df_memory.iat[idx, field_map['delta_t']]
                    df_memory.iat[idx, field_map['t_history']] += f',{ivl}'

                    halflife = df_memory.iat[idx, field_map['halflife']]
                    difficulty = df_memory.iat[idx, field_map['difficulty']]
                    p_recall = df_memory.iat[idx, field_map['p_recall']]
                    df_memory.iat[idx, field_map['p_history']] += f',{p_recall:.2f}'
                    reps = df_memory.iat[idx, field_map['reps']]
                    lapses = df_memory.iat[idx, field_map['lapses']]
                    state = df_memory.iat[idx, field_map['state']]

                    if random.random() < p_recall:
                        day_cost += recall_cost

                        df_memory.iat[idx, field_map['r_history']] += ',1'

                        new_state, new_halflife = student.next_state(state, 1, ivl, p_recall)
                        df_memory.iat[idx, field_map['halflife']] = new_halflife
                        df_memory.iat[idx, field_map['state']] = new_state
                        df_memory.iat[idx, field_map['reps']] = reps + 1
                        df_memory.iat[idx, field_map['cost']] += recall_cost

                        if new_halflife >= target_halflife:
                            meet_target += 1
                            df_memory.iat[idx, field_map['halflife']] = np.inf
                            df_memory.iat[idx, field_map['due_date']] = np.inf
                            continue

                        delta_t = scheduler(df_memory.loc[idx], method)
                        df_memory.iat[idx, field_map['due_date']] = day + delta_t

                    else:
                        day_cost += forget_cost

                        df_memory.iat[idx, field_map['r_history']] += ',0'

                        new_state, new_halflife = student.next_state(state, 0, ivl, p_recall)

                        if new_halflife >= target_halflife:
                            meet_target += 1
                            df_memory.iat[idx, field_map['halflife']] = np.inf
                            df_memory.iat[idx, field_map['due_date']] = np.inf
                            continue

                        df_memory.iat[idx, field_map['halflife']] = new_halflife
                        df_memory.iat[idx, field_map['state']] = new_state

                        reps = 0
                        lapses = lapses + 1

                        df_memory.iat[idx, field_map['reps']] = reps
                        df_memory.iat[idx, field_map['lapses']] = lapses
                        df_memory.iat[idx, field_map['cost']] += forget_cost

                        delta_t = scheduler(df_memory.loc[idx], method)
                        df_memory.iat[idx, field_map['due_date']] = day + delta_t
                        df_memory.iat[idx, field_map['cost']] += recall_cost

                need_learn = df_memory[df_memory['halflife'].isna()].index

                for idx in need_learn:
                    if day_cost >= day_cost_limit:
                        break
                    learned += 1
                    day_cost += new_cost
                    df_memory.iat[idx, field_map['last_date']] = day

                    difficulty = df_memory.iat[idx, field_map['difficulty']]
                    reps = df_memory.iat[idx, field_map['reps']]
                    lapses = df_memory.iat[idx, field_map['lapses']]

                    r, t, p, new_state, new_halflife = student.init(difficulty)

                    df_memory.iat[idx, field_map['r_history']] = str(r)
                    df_memory.iat[idx, field_map['t_history']] = str(t)
                    df_memory.iat[idx, field_map['p_history']] = str(p)
                    df_memory.iat[idx, field_map['halflife']] = new_halflife
                    df_memory.iat[idx, field_map['state']] = new_state

                    delta_t = scheduler(df_memory.loc[idx], method)
                    df_memory.iat[idx, field_map['due_date']] = day + delta_t
                    df_memory.iat[idx, field_map['cost']] = 0

                new_item_per_day[day] = learned
                learned_per_day[day] = learned_per_day[day - 1] + learned
                cost_per_day[day] = day_cost

                if day >= period_len:
                    new_item_per_day_average_per_period[day] = np.true_divide(new_item_per_day[day - period_len:day].sum(),
                                                                              period_len)
                    cost_per_day_average_per_period[day] = np.true_divide(cost_per_day[day - period_len:day].sum(),
                                                                          period_len)
                else:
                    new_item_per_day_average_per_period[day] = np.true_divide(new_item_per_day[:day + 1].sum(), day + 1)
                    cost_per_day_average_per_period[day] = np.true_divide(cost_per_day[:day + 1].sum(), day + 1)

                record_per_day[day] = df_memory['p_recall'].sum()
                meet_target_per_day[day] = meet_target

            total_learned = int(sum(new_item_per_day))
            total_cost = int(sum(cost_per_day))

            plt.figure(1)
            plt.plot(record_per_day, label=f'{method}')

            plt.figure(2)
            plt.plot(meet_target_per_day, label=f'{method}')
            cost_day = np.argmax(meet_target_per_day >= compare_target)
            if cost_day > 0:
                print(f'cost day: {cost_day}')
                plt.plot(cost_day, compare_target, 'k*', linewidth=2)

            plt.figure(3)
            plt.plot(new_item_per_day_average_per_period, label=f'{method}')

            plt.figure(4)
            plt.plot(cost_per_day_average_per_period, label=f'{method}')

            plt.figure(5)
            plt.plot(learned_per_day, label=f'{method}')

            print('acc learn', total_learned)
            print('meet target', meet_target)

            save = df_memory[df_memory['p_recall'] > 0].copy()
            save['halflife'] = round(save['halflife'], 4)
            save['p_recall'] = round(save['p_recall'], 4)

            save.to_csv(f'./simulation/{method}.tsv', index=False, sep='\t')

        plt.figure(1)
        # plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
        plt.xlabel("days")
        plt.ylabel("summation of recall probability")
        # plt.legend()
        plt.grid(True)
        plt.savefig(f'./plot/{model}_SRP.eps', bbox_inches='tight')
        plt.figure(2)
        plt.plot((0, learn_days), (compare_target, compare_target), color='black', linestyle='dotted')
        # plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
        plt.xlabel("days")
        plt.ylabel("target half-life reached")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./plot/{model}_THR.eps', bbox_inches='tight')
        plt.figure(3)
        plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
        plt.xlabel("days")
        plt.ylabel(f"new item per day({period_len} days average)")
        plt.grid(True)
        plt.savefig(f'./plot/{model}_NEW.eps', bbox_inches='tight')
        plt.figure(4)
        plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
        plt.xlabel("days")
        plt.ylabel(f"cost per day({period_len} days average)")
        # plt.legend()
        plt.grid(True)
        plt.savefig(f'./plot/{model}_COST.eps', bbox_inches='tight')
        plt.figure(5)
        # plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
        plt.xlabel("days")
        plt.ylabel(f"items total learned")
        # plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f'./plot/{model}_ITL.eps', bbox_inches='tight')
        plt.close('all')
