import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


models = ('HLR', 'HLR-lex', 'DHP', 'leitner', 'pimsleur', 'nn-GRU_nh-2_loss-sMAPE', 'nn-GRU_nh-2_loss-sMAPE-p',
          'nn-GRU_nh-2_loss-sMAPE-t', 'nn-GRU_nh-2_loss-sMAPE-p-t')


# models = ('pimsleur', 'leitner', 'HLR-lex', 'HLR', 'nn-GRU_nh-4_loss-sMAPE-p-t', 'nn-GRU_nh-4_loss-sMAPE-p',
# 'nn-GRU_nh-4_loss-sMAPE-t', 'nn-GRU_nh-4_loss-sMAPE')

def smape(A, F):
    return 1 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


for m in models:
    print(f'model: {m}')
    result_files = os.listdir(f'./result/{m}')
    if '.DS_Store' in result_files:
        result_files.remove('.DS_Store')
    if len(result_files) == 0:
        continue
    result_files.sort()
    avg_mae = []
    avg_mse = []
    avg_mape = []
    avg_smape = []
    avg_mae_h = []
    for i, filename in enumerate(result_files):
        data = pd.read_csv(f'./result/{m}/{filename}', sep='\t', index_col=[0])
        avg_mae.append(mean_absolute_error(data['p'], data['pp']))
        avg_mse.append(mean_squared_error(data['p'], data['pp']))
        avg_mape.append(mean_absolute_percentage_error(data['h'], data['hh']))
        avg_smape.append(smape(data['h'], data['hh']))
        avg_mae_h.append(mean_absolute_error(data['h'], data['hh']))
        print(
            f"{filename}\tmae(p): {avg_mae[i]:.4f}\tmse(p): {avg_mse[i]:.4f}\tmape(h): {avg_mape[i]:.4f}\tsmape(h): {avg_smape[i]:.4f}\tmae(h): {avg_mae_h[i]:.4f}")
    print(
        f"avg\tmae(p): {sum(avg_mae) / len(avg_mae):.4f}\tmse(p): {sum(avg_mse) / len(avg_mse):.4f}\tmape(h): {sum(avg_mape) / len(avg_mape):.4f}\tsmape(h): {sum(avg_smape) / len(avg_smape):.4f}\tmae(h): {sum(avg_mae_h) / len(avg_mae_h):.4f}")
