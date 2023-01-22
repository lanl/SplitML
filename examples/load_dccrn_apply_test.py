"""
Load DCCRN-DR models
and apply to test and crossover-test data.

"""

 # %%
import os
import torch
import numpy as np
import pandas as pd
from inspect import getmembers, isfunction

from splitml import losses

losses_dict = {k: v for (k, v) in getmembers(losses, isfunction)}
losses_dict['mse'] = torch.nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nt = 1024
synth_data_path = '/data/synth_data/'

# %% Load results from info files

model_files = os.listdir('saved_models/')
model_dates = ['_'.join(c.split('_')[2:4]) for c in model_files if 'DCCRN_DR' in c]

args_list = []
for d in model_dates:
    a = torch.load('saved_models/DCCRN_DR_%s_info.pt' % d)
    a['date'] = d 
    args_list.append(a)

args_merged = {}
for i in args_list:
    for k,v in i.items():
        if k in args_merged:
            args_merged[k].append(v)
        else:
            args_merged[k] = [v]

args_merged = pd.DataFrame(args_merged)

# %% create keys for each data set wit data frame describing, load test data into dicts by keys
test_data_info = {'test_key': [], 'snr': [], 'fc': [], 'noisedist': []}
test_sigs_dict = {}
test_noise_dict = {}
test_A_dict = {}
data_keys = ['%d' % d for d in range(50)]

raw_data = {'f0_high': np.load('%s/synthetic_data_highSNR_0.npz' % synth_data_path),
            'f1600_high': np.load('%s/synthetic_data_highSNR_1600.npz' % synth_data_path),
            'f0_low': np.load('%s/synthetic_data_lowSNR_0.npz' % synth_data_path),
            'f1600_low': np.load('%s/synthetic_data_lowSNR_1600.npz' % synth_data_path)
}

for kw, fc in zip(['0', 'peak'], ['f0', 'f1600']):
    for snr in ['low', 'high']:
        test_sigs = raw_data['%s_%s' % (fc, snr)]['clean_testing_%s' % kw]
        test_A = raw_data['%s_%s' % (fc, snr)]['df_testing_%s' % kw][:, 3]

        for nkw, noisedist in zip(['white', 'sampled'], ['whitenoise', 'realnoise']):
            test_noise = raw_data['%s_%s' % (fc, snr)]['%s_noise_testing_%s' % (nkw, kw)][:2000, :] 

            dk = data_keys.pop(0)
            test_sigs_dict[dk] = test_sigs
            test_noise_dict[dk] = test_noise
            test_A_dict[dk] = test_A

            test_data_info['test_key'].append(dk)
            test_data_info['snr'].append(snr)
            test_data_info['fc'].append(fc)
            test_data_info['noisedist'].append(noisedist)

test_data_info = pd.DataFrame(test_data_info)

# %% Associate each model with its train/test dataset
args_merged['data_key'] = np.nan
for i in test_data_info.iterrows():
    dk = i[1]['test_key']
    hsnr = True if i[1]['snr'] == 'high' else False
    cf0 = True if i[1]['fc'] == 'f0' else False
    rn = True if i[1]['noisedist'] == 'realnoise' else False
    crit ='(high_snr_data == @hsnr) & (central_f_zero == @cf0) & (use_real_noise == @rn)'
    args_merged.loc[args_merged.eval(crit), 'data_key'] = dk

# %% Calculate Rsq for comparison to other results
args_merged['test_rsq'] = np.nan
for dk in np.unique(args_merged['data_key']):
    A_vals = test_A_dict[dk]
    ts = test_sigs_dict[dk]/A_vals[:, None]
    denom = np.mean(ts * np.conj(ts)).real
    args_merged.loc[args_merged['data_key']==dk, 'test_rsq'] = 1.0 - args_merged.loc[args_merged['data_key']==dk, 'test_mse'].values/denom

# %% Create easier model key
def categorise(row):  
    # all true
    if row['central_f_zero'] == True and row['high_snr_data'] == True and row['use_real_noise'] == True:
        return 'fc0_high_snr_realnoise'
    # two true
    elif row['central_f_zero'] == True and row['high_snr_data'] == True and row['use_real_noise'] == False:
        return 'fc0_high_snr_whitenoise'
    elif row['central_f_zero'] == True and row['high_snr_data'] == False and row['use_real_noise'] == True:
        return 'fc0_low_snr_realnoise'
    elif row['central_f_zero'] == False and row['high_snr_data'] == True and row['use_real_noise'] == True:
        return 'fc1600_high_snr_realnoise'
    # one true
    elif row['central_f_zero'] == True and row['high_snr_data'] == False and row['use_real_noise'] == False:
        return 'fc0_low_snr_whitenoise'
    elif row['central_f_zero'] == False and row['high_snr_data'] == True and row['use_real_noise'] == False:
        return 'fc1600_high_snr_whitenoise'
    elif row['central_f_zero'] == False and row['high_snr_data'] == False and row['use_real_noise'] == True:
        return 'fc1600_low_snr_realnoise'
    # zero true
    elif row['central_f_zero'] == False and row['high_snr_data'] == False and row['use_real_noise'] == False:
        return 'fc1600_low_snr_whitenoise'
    else: 
        return 'what'
    
args_merged['experiment'] = args_merged.apply(lambda row: categorise(row), axis=1)

# %%
args_merged.to_csv('saved_models/dccrn_results.csv')
# %%
