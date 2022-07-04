import argparse
import random
import sys
import numpy as np
import pandas as pd
import math
from collections import namedtuple
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold

Instance = namedtuple('Instance', 'p t fv h a r_history t_history p_history'.split())
duolingo_algo = ('HLR', 'LR', 'leitner', 'pimsleur')
rnn_algo = ('GRU', 'RNN', 'LSTM')


def load_data(input_file):
    dataset = pd.read_csv(input_file, sep='\t', index_col=None)
    dataset = dataset[dataset['halflife'] > 0]
    dataset = dataset[dataset['i'] > 0]
    dataset = dataset[
        dataset['p_history'].map(lambda x: len(x.split(','))) == dataset['t_history'].map(lambda x: len(x.split(',')))]
    # dataset.drop_duplicates(subset=['r_history', 't_history', 'p_history', 'difficulty'], inplace=True)
    # dataset['weight'] = dataset['total_cnt'] / dataset['total_cnt'].sum()
    # dataset['weight'] = dataset['total_cnt'] / dataset['total_cnt'].sum()
    # std = preprocessing.MinMaxScaler()
    # dataset['weight_std'] = std.fit_transform(dataset[['weight']]) + 1
    dataset['weight_std'] = 1
    return dataset


def feature_extract(train_set, test_set, method, omit_lexemes=False):
    instances = {'train': [], 'test': []}
    for set_id, data in (('train', train_set), ('test', test_set)):
        for i, row in data.iterrows():
            p = float(row['p_recall'])
            t = max(1, int(row['delta_t']))
            h = float(row['halflife'])
            right = row['r_history'].count('1')
            wrong = row['r_history'].count('0')
            total = right + wrong
            # feature vector is a list of (feature, value) tuples
            fv = []
            # core features based on method
            # optional flag features
            if method == 'pimsleur':
                fv.append((sys.intern('total'), right + wrong))
            elif method == 'leitner':
                fv.append((sys.intern('diff'), right - wrong))
            else:
                fv.append((sys.intern('right'), math.sqrt(1 + right)))
                fv.append((sys.intern('wrong'), math.sqrt(1 + wrong)))

            if method == 'LR':
                fv.append((sys.intern('time'), t))
            if not omit_lexemes:
                fv.append((sys.intern('%s' % (row['d'])), 1.))

            fv.append((sys.intern('bias'), 1.))
            instances[set_id].append(
                Instance(p, t, fv, h, (right + 2.) / (total + 4.), row['r_history'], row['t_history'],
                         row['p_history']))
            if i % 1000000 == 0:
                sys.stderr.write('%d...' % i)
        sys.stderr.write('done!\n')
    return instances['train'], instances['test']


argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('-l', action="store_true", default=False, help='omit lexeme features')
argparser.add_argument('-p', action="store_true", default=False, help='omit p history features')
argparser.add_argument('-t', action="store_true", default=False, help='omit t history features')
argparser.add_argument('-test', action="store_true", default=False, help='test model')
argparser.add_argument('-train', action="store_true", default=False, help='train model')
argparser.add_argument('-m', action="store", dest="method", default='GRU', help="LSTM, HLR, LR, SM2")
argparser.add_argument('-hidden', action="store", dest="h", default='16', help="4, 8, 16, 32")
argparser.add_argument('-loss', action="store", dest="loss", default='MAPE', help="MAPE, L1, MSE, sMAPE")
argparser.add_argument('input_file', action="store", help='log file for training')

if __name__ == "__main__":

    random.seed(2022)
    args = argparser.parse_args()
    sys.stderr.write('method = "%s"\n' % args.method)
    if args.l:
        sys.stderr.write('--> omit_lexemes\n')
    if args.p:
        sys.stderr.write('--> omit_p_history\n')
    if args.t:
        sys.stderr.write('--> omit_t_history\n')
    sys.stderr.write(f'{args.h} --> n_hidden\n')
    sys.stderr.write(f'{args.loss} --> loss\n')

    dataset = load_data(args.input_file)
    test = dataset.sample(frac=0.8, random_state=2022)
    train = dataset.drop(index=test.index)
    if not args.train:
        if not args.test:
            train_train, train_test = train_test_split(train, test_size=0.5, random_state=2022)
            sys.stderr.write('|train| = %d\n' % len(train_train))
            sys.stderr.write('|test|  = %d\n' % len(train_test))
            if args.method in rnn_algo:
                from model.RNN_HLR import SpacedRepetitionModel

                model = SpacedRepetitionModel(train_train, train_test, omit_p_history=args.p, omit_t_history=args.t,
                                              hidden_nums=int(args.h), loss=args.loss, network=args.method)
                model.train()
                model.eval(0, 0)
            elif args.method in duolingo_algo:
                from model.halflife_regression import SpacedRepetitionModel

                train_fold, test_fold = feature_extract(train_train, train_test, args.method, args.l)
                model = SpacedRepetitionModel(train_fold, test_fold, method=args.method)
                model.train()
                model.eval(0, 0)
            elif args.method == 'DHP':
                from model.DHP import SpacedRepetitionModel

                model = SpacedRepetitionModel(train_train, train_test)
                model.train()
                model.eval(0, 0)
        else:
            # kf = KFold(n_splits=5, shuffle=True, random_state=2022)
            kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=2022)
            for idx, (train_index, test_fold) in enumerate(kf.split(test)):
                train_fold = dataset.iloc[train_index]
                test_fold = dataset.iloc[test_fold]
                repeat = idx // 2 + 1
                fold = idx % 2 + 1
                sys.stderr.write('Repeat %d, Fold %d\n' % (repeat, fold))
                sys.stderr.write('|train| = %d\n' % len(train_index))
                sys.stderr.write('|test|  = %d\n' % len(test_fold))
                if args.method in rnn_algo:
                    from model.RNN_HLR import SpacedRepetitionModel

                    model = SpacedRepetitionModel(train_fold, test_fold, omit_p_history=args.p, omit_t_history=args.t,
                                                  hidden_nums=int(args.h), loss=args.loss, network=args.method)
                    model.train()
                    model.eval(repeat, fold)
                elif args.method in duolingo_algo:
                    from model.halflife_regression import SpacedRepetitionModel

                    train_fold, test_fold = feature_extract(train_fold, test_fold, args.method, args.l)
                    model = SpacedRepetitionModel(train_fold, test_fold, method=args.method, omit_lexemes=args.l)
                    model.train()
                    model.eval(repeat, fold)
                elif args.method == 'SM2':
                    from model.SM2 import eval

                    eval(test_fold, repeat, fold)
                elif args.method == 'DHP':
                    from model.DHP import SpacedRepetitionModel

                    model = SpacedRepetitionModel(train_fold, test_fold)
                    model.train()
                    model.eval(repeat, fold)
                else:
                    break
            test['pp'] = test['p_recall'].mean()
            print(test['p_recall'].mean())
            test['mae(p)'] = abs(test['pp'] - test['p_recall'])
            print("mae(p)", test['mae(p)'].mean())
            test['hh'] = np.log(test['pp']) / np.log(test['p_recall']) * test['delta_t']
            test['MAPE(h)'] = abs((test['hh'] - test['halflife']) / test['halflife'])
            print("MAPE(h)", test['MAPE(h)'].mean())
    else:
        # train_train, train_test = train_test_split(dataset, test_size=0.2, random_state=2022)
        sys.stderr.write('|train| = %d\n' % len(dataset))
        if args.method in rnn_algo:
            from model.RNN_HLR import SpacedRepetitionModel

            model = SpacedRepetitionModel(dataset, dataset, omit_p_history=args.p, omit_t_history=args.t,
                                          hidden_nums=int(args.h), loss=args.loss, network=args.method)
            model.train()
        elif args.method in duolingo_algo:
            from model.halflife_regression import SpacedRepetitionModel

            train_fold, test_fold = feature_extract(dataset, dataset, args.method, args.l)
            model = SpacedRepetitionModel(train_fold, test_fold, method=args.method, omit_lexemes=args.l)
            model.train()
        elif args.method == 'DHP':
            from model.DHP import SpacedRepetitionModel

            model = SpacedRepetitionModel(dataset, dataset)
            model.train()