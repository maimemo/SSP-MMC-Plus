import pandas as pd
import numpy as np
from tqdm import tqdm

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

def halflife_forgetting_curve(x, h):
    return np.power(2, - x / h)


def cal_halflife(group):
    if group['i'].values[0] > 1:
        r_ivl_cnt = sum(group['delta_t'] * group['p_recall'].map(np.log) * group['total_cnt'])
        ivl_ivl_cnt = sum(group['delta_t'].map(lambda x: x ** 2) * group['total_cnt'])
        group['halflife'] = round(np.log(0.5) / (r_ivl_cnt / ivl_ivl_cnt), 4)
    else:
        group['halflife'] = 0.0
    group['group_cnt'] = sum(group['total_cnt'])
    return group


if __name__ == "__main__":
    data = pd.read_csv('./data/opensource_dataset_forgetting_curve.tsv', sep='\t', index_col=None)
    data = data[(data['p_recall'] < 1) & (data['p_recall'] > 0)]
    data = data.groupby(
        by=['d', 'i', 'r_history', 't_history']).apply(
        cal_halflife)

    data['p_recall'] = data['p_recall'].map(lambda x: round(x, 2))
    data['p_history'] = '0'
    data.sort_values('i', inplace=True)
    data.to_csv('./data/opensource_dataset_halflife.tsv', sep='\t', index=None)
    data = pd.read_csv('./data/opensource_dataset_halflife.tsv', sep='\t', index_col=None)

    for idx in tqdm(data[(data['i'] == 2)].index):
        data.loc[idx, 'p_history'] = d2p[data.loc[idx, 'd']-1]

    data['p_history'] = data['p_history'].map(lambda x: str(x))

    for idx in tqdm(data[data['i'] >= 2].index):
        item = data.loc[idx]
        interval = int(item['delta_t'])
        index = data[(data['r_history'].str.startswith(item['r_history'])) & (
                data['t_history'] == item['t_history'] + f',{interval}') & (
                             data['d'] == item['d'])].index
        data.loc[index, 'p_history'] = item['p_history'] + ',' + str(item['p_recall'])
        data.loc[index, 'last_halflife'] = item['halflife']
        data.loc[index, 'last_p_recall'] = item['p_recall']

    data.to_csv('./data/opensource_dataset_p_history.tsv', sep='\t', index=None)