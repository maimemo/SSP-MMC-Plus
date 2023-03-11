import os
from pathlib import Path

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# plt.rcParams['figure.dpi'] = 900
models = ('nn-GRU_nh-2_loss-sMAPE', 'nn-GRU_nh-2_loss-sMAPE-p',
          'nn-GRU_nh-2_loss-sMAPE-t', 'nn-GRU_nh-2_loss-sMAPE-p-t', 'DHP', 'HLR', 'HLR-lex', 'pimsleur', 'leitner')
plt.style.use('seaborn-whitegrid')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams['figure.figsize'] = (8.0, 16.0)
plt.rcParams.update({'font.size': 24})


def load_brier(predictions, real, bins=20):
    counts = np.ones(bins)
    correct = np.zeros(bins)
    prediction = np.zeros(bins)
    for p, r in zip(predictions, real):
        bin = min(int(p * bins), bins - 1)
        counts[bin] += 1
        correct[bin] += r
        prediction[bin] += p
    prediction_means = prediction / counts
    prediction_means[np.isnan(prediction_means)] = ((np.arange(bins) + 0.5) / bins)[np.isnan(prediction_means)]
    correct_means = correct / counts
    size = len(predictions)
    answer_mean = sum(correct) / size
    return {
        "reliability": sum(counts * (correct_means - prediction_means) ** 2) / size,
        "resolution": sum(counts * (correct_means - answer_mean) ** 2) / size,
        "uncertainty": answer_mean * (1 - answer_mean),
        "detail": {
            "bin_count": bins,
            "bin_counts": list(counts),
            "bin_prediction_means": list(prediction_means),
            "bin_correct_means": list(correct_means),
        }
    }


def plot_brier(predictions, real, bins=20):
    brier = load_brier(predictions, real, bins=bins)
    bin_count = brier['detail']['bin_count']
    counts = np.array(brier['detail']['bin_counts'])
    bins = (np.arange(bin_count) + 0.5) / bin_count
    plt.figure()
    plt.ylabel('Number of predictions')
    plt.bar(bins, counts, width=(0.5 / bin_count), color='white', edgecolor="black", label='Number of predictions')
    plt.legend(loc='lower center')
    plt.twinx()
    plt.plot((0, 1), (0, 1), '--', color='black', label='Optimal average observation')
    plt.plot(brier['detail']['bin_prediction_means'], brier['detail']['bin_correct_means'], '*', color='black',
             label='Average observation')
    plt.xlabel('Prediction')
    plt.ylabel('Observeation')
    plt.legend(loc='upper center')


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


if __name__ == "__main__":
    data = pd.read_csv('./data/dataset_trans.csv')
    Path('./plot').mkdir(parents=True, exist_ok=True)
    plt.hist(data['p'], range=(0, 1), width=(1 / 20), bins=20, color='white', edgecolor='black')
    plt.ylabel("Number of samples", fontsize=24)
    plt.xlabel("P(recall)", fontsize=24)
    plt.savefig(f'./plot/p_distribution.eps')
    plt.close()

    plt.hist(data['h'], width=(data['h'].max() / 20), bins=20, color='white', edgecolor='black', log=True)
    plt.ylabel("Number of samples", fontsize=24)
    plt.xlabel("half-life (days)", fontsize=24)
    plt.savefig(f'./plot/h_distribution.eps')
    plt.close()

    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    flag = True
    lns = []

    pfig = plt.figure(2)
    pax1 = pfig.add_subplot(111)
    pax2 = pax1.twinx()
    plns = []

    for m in models:
        print(f'model: {m}')
        result_files = os.listdir(f'./result/{m}')
        if '.DS_Store' in result_files:
            result_files.remove('.DS_Store')
        if len(result_files) == 0:
            continue
        p_df = pd.DataFrame(columns=['p', 'ae'])
        h_df = pd.DataFrame(columns=['h', 'mae_h', 'smape_h'])
        for i, filename in enumerate(result_files):
            data = pd.read_csv(f'./result/{m}/{filename}', sep='\t', index_col=[0])
            data['mae_h'] = abs(data['h'] - data['hh'])
            data['smape_h'] = abs(data['h'] - data['hh']) / (abs(data['h']) + abs(data['hh'])) * 2
            p_df = p_df.append(data[['p', 'ae']])
            h_df = h_df.append(data[['h', 'mae_h', 'smape_h']])
        print("smape_h: ", h_df['smape_h'].mean())
        h_df['h_bin'] = h_df['h'].map(lambda x: math.pow(1.2, round(math.log(x + 1.2, 1.2))))
        h_group = h_df.groupby(by='h_bin').count()

        m = m.replace('nn-', '')
        m = m.replace('_nh-2_loss-sMAPE', '')
        m = m.replace('GRU', 'GRU-HLR')
        m = m.replace('-p', ' -p')
        m = m.replace('-t', ' -t')
        m = m.replace('-lex', ' -lex')

        if flag:
            lns1 = ax1.bar(x=h_group.index, height=h_group['smape_h'], width=h_group.index / 5.5,
                           ec='k', lw=.2, label='Number of samples', color='white')
            ax1.set_ylabel("Number of samples")
            ax1.set_xlabel("Half-life (days)")
            ax1.semilogx()
            lns.append(lns1)

        h_group = h_df.groupby(by='h_bin').agg('mean')
        lns2 = ax2.plot(h_group['smape_h'], label=f'{m}')
        ax2.set_ylabel("Symmetric Mean Absolute Percentage Error")
        lns.append(lns2[0])

        p_df['p_bin'] = p_df['p'].map(lambda x: round(4 * x, 1) / 4)
        p_group = p_df.groupby(by='p_bin').count()

        if flag:
            plns1 = pax1.bar(x=p_group.index, height=p_group['ae'], width=0.025,
                             ec='k', lw=.2, label='Number of samples', color='white')
            pax1.set_ylabel("Number of samples")
            pax1.set_xlabel("Probability of recall")
            plns.append(plns1)
            flag = False

        p_group = p_df.groupby(by='p_bin').agg('mean')
        plns2 = pax2.plot(p_group['ae'], label=f'{m}')
        pax2.set_ylabel("Mean Absolute Error")
        plns.append(plns2[0])

    plt.figure(1)
    labs = [l.get_label() for l in lns]
    # ax2.legend(lns, labs)
    plt.grid(linestyle='--')
    # plt.title(f'{m}')
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
    # plt.show()
    start, end = plt.gca().get_xlim()
    plt.gca().xaxis.set_ticks(np.round(np.power(4, np.arange(np.log(start)/np.log(4), np.log(end)/np.log(4), 1))))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.savefig(f'./plot/smape-distribution.eps', bbox_inches='tight')

    plt.close()

    plt.figure(2)
    plabs = [l.get_label() for l in plns]
    pax2.legend(plns, plabs, loc='upper left', bbox_to_anchor=(0.18, 0., 0., 1.))
    plt.grid(linestyle='--')
    # plt.title(f'{m}')
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.show()
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
    plt.savefig(f'./plot/mae-distribution.eps', bbox_inches='tight')
    plt.close()

    # plot_brier(p_df['p'], p_df['pp'], 20)
    # plt.title(f'{m}')
    # plt.savefig(f'./plot/{m}-mae-distribution.eps')
    # plt.close()
    # print('\n')
