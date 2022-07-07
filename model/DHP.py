import time
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from model.utils import *


class SpacedRepetitionModel(object):
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.ra = 0
        self.rb = 0
        self.rc = 0
        self.rd = 0
        self.fa = 0
        self.fb = 0
        self.fc = 0
        self.fd = 0

    def train(self):
        tmp = self.train_set.copy()
        tmp['halflife_increase'] = round(tmp['halflife'] / tmp['last_halflife'], 4)
        tmp = tmp[tmp['i'] > 2]
        tmp['last_recall'] = tmp['r_history'].map(lambda x: x[-1])
        del tmp['delta_t']
        del tmp['p_recall']
        del tmp['total_cnt']
        tmp.drop_duplicates(inplace=True)
        tmp.dropna(inplace=True)
        self.fit_recall_halflife(tmp[(tmp['halflife_increase'] > 1) & (tmp['last_recall'] == '1') & (
                tmp['r_history'].str.count('0') == 1)].copy())
        self.fit_forget_halflife(tmp[(tmp['last_recall'] == '0') & (
                tmp['r_history'].str.count('0') == 2)].copy())
        self.save()

    def fit_recall_halflife(self, raw):
        print('fit_recall_halflife_dhp')
        raw['log_hinc'] = raw['halflife_increase'].map(lambda x: np.log(x - 1))
        raw['log_h'] = raw['last_halflife'].map(lambda x: np.log(x))
        raw['fi'] = raw['last_p_recall'].map(lambda x: 1 - x)
        raw['log_fi'] = raw['last_p_recall'].map(lambda x: np.log(1 - x))
        raw['log_d'] = raw['d'].map(lambda x: np.log(x))
        raw['log_delta_h'] = np.log(raw['halflife'] - raw['last_halflife'])
        # print(raw.corr())

        X = raw[['log_d', 'log_h', 'log_fi']]
        Y = raw[['log_hinc']]

        lr = LinearRegression()
        lr.fit(X, Y)
        print('Intercept: ', lr.intercept_)
        print('Slope: ', lr.coef_)
        self.ra = lr.intercept_[0]
        self.rb = lr.coef_[0][0]
        self.rc = lr.coef_[0][1]
        self.rd = lr.coef_[0][2]

    def fit_forget_halflife(self, raw):
        print('fit_forget_halflife_dhp')
        raw['log_h'] = raw['last_halflife'].map(lambda x: np.log(x))
        raw['fi'] = raw['last_p_recall'].map(lambda x: 1 - x)
        raw['log_fi'] = raw['last_p_recall'].map(lambda x: np.log(1 - x))
        raw['log_d'] = raw['d'].map(lambda x: np.log(x))
        raw['log_halflife'] = raw['halflife'].map(lambda x: np.log(x))

        # print(raw.corr())

        X = raw[['log_d', 'log_h', 'log_fi']]
        Y = raw[['log_halflife']]

        lr = LinearRegression()
        lr.fit(X, Y)
        print('Intercept: ', lr.intercept_)
        print('Slope: ', lr.coef_)
        self.fa = lr.intercept_[0]
        self.fb = lr.coef_[0][0]
        self.fc = lr.coef_[0][1]
        self.fd = lr.coef_[0][2]

    def cal_start_halflife(self, d):
        return - 1 / np.log2(max(0.925 - 0.05 * d, 0.025))

    def cal_recall_halflife(self, d, halflife, p_recall):
        return halflife * (
                1 + np.exp(self.ra) * np.power(d, self.rb) * np.power(halflife, self.rc) * np.power(
            1 - p_recall, self.rd))

    def cal_forget_halflife(self, d, halflife, p_recall):
        return np.exp(self.fa) * np.power(d, self.fb) * np.power(halflife, self.fc) * np.power(
            1 - p_recall, self.fd)

    def dhp(self, line, h, d):
        recall = int(line[0])
        interval = int(line[1])
        if recall == 1:
            if interval == 0:
                h = self.cal_start_halflife(d)
            else:
                p_recall = np.exp2(- interval / h)
                h = self.cal_recall_halflife(d, h, p_recall)
        else:
            if interval == 0:
                h = self.cal_start_halflife(d)
            else:
                p_recall = np.exp2(- interval / h)
                h = self.cal_forget_halflife(d, h, p_recall)
                d = min(d + 2, 18)
        return h, d

    def eval(self, repeat, fold):
        record = pd.DataFrame(
            columns=['r_history', 't_history', 'p_history',
                     't',
                     'h', 'hh', 'p', 'pp', 'ae', 'ape'])
        p_loss = 0
        h_loss = 0
        count = 0
        for idx, line in tqdm(self.test_set.iterrows(), total=self.test_set.shape[0]):
            line_tensor = lineToTensor(list(
                zip([line['r_history']], [line['t_history']], [line['p_history']]))[0])
            ph = 0
            d = line['d']
            for j in range(line_tensor.size()[0]):
                ph, d = self.dhp(line_tensor[j][0], ph, d)

            # print(f'model: {m}\tsample: {line}\tcorrect: {interval}\tpredict: {float(output)}')

            pp = np.power(2, -line['delta_t'] / ph)
            p = line['p_recall']
            p_loss += abs(p - pp)

            h = line['halflife']
            h_loss += abs((ph - h) / h)
            count += 1
            record = pd.concat([record, pd.DataFrame(
                {'r_history': [line['r_history']],
                 't_history': [line['t_history']],
                 'p_history': [line['p_history']],
                 't': [line['delta_t']], 'h': [line['halflife']],
                 'hh': [round(ph, 2)], 'p': [p],
                 'pp': [round(pp, 3)], 'ae': [round(abs(p - pp), 3)],
                 'ape': [round(abs(ph - h) / h, 3)]})],
                               ignore_index=True)
        print(f"model: DHP")
        print(f'sample num: {count}')
        print(f"avg p loss: {p_loss / count}")
        print(f"avg h loss: {h_loss / count}")
        Path(f'./result/DHP').mkdir(parents=True, exist_ok=True)
        record.to_csv(f'./result/DHP/repeat{repeat}_fold{fold}_{int(time.time())}.tsv', sep='\t', index=False)

    def save(self):
        Path('./tmp/DHP').mkdir(parents=True, exist_ok=True)
        parameters = pd.DataFrame.from_dict(
            {'ra': [self.ra], 'rb': [self.rb], 'rc': [self.rc], 'rd': [self.rd], 'fa': [self.fa], 'fb': [self.fb],
             'fc': [self.fc],
             'fd': [self.fd]})
        parameters.to_csv('./tmp/DHP/model.csv', index=False)
