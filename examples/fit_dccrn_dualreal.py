"""
Script to fit DCCRN on synthetic NQR data.
DualReal approach based on concatenation of real and imaginary parts.

Author: Natalie Klein
"""
import os
from inspect import getmembers, isfunction
import argparse
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from splitml import losses

from scripts.dccrn.dc_crn import DCCRN

losses_dict = {k: v for (k, v) in getmembers(losses, isfunction)}
losses_dict['mse'] = torch.nn.MSELoss()

def main():

    if args.use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # %% Settings 
    model_fn = DCCRN
    seed = args.seed
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.n_epochs
    loss_fn = losses_dict[args.loss_fn]
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    nt = 1024
    t = np.linspace(0, 0.018414, nt)

    save_stub = 'DCCRN_DR_%s' % datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    # %% Load noise data, split train/test
    with open(args.real_noise_path, 'rb') as f:
        t, data, _ = pickle.load(f)
    data = np.array(data) # (4096, 16384)
    t = t[:nt]
    if args.high_snr_data:
        if args.central_f_zero:
            data = np.load('%s/synthetic_data_highSNR_0.npz' % args.synth_data_path)
        else:
            data = np.load('%s/synthetic_data_highSNR_1600.npz' % args.synth_data_path)
    else:
        if args.central_f_zero:
            data = np.load('%s/synthetic_data_lowSNR_0.npz' % args.synth_data_path)
        else:
            data = np.load('%s/synthetic_data_lowSNR_1600.npz' % args.synth_data_path)
    if args.central_f_zero:
        kw = '0'
    else:
        kw = 'peak'
    train_sigs = data['clean_training_%s' % kw]
    val_sigs = data['clean_validation_%s' % kw]
    test_sigs = data['clean_testing_%s' % kw]
    test_A = data['df_testing_%s' % kw][:, 3]
    if args.use_real_noise:
        train_noise = data['sampled_noise_training_%s' % kw]
        val_noise = data['sampled_noise_validation_%s' % kw]
        test_noise = data['sampled_noise_testing_%s' % kw][:2000, :] 
    else:
        train_noise = data['white_noise_training_%s' % kw]
        val_noise = data['white_noise_validation_%s' % kw]
        test_noise = data['white_noise_testing_%s' % kw]


    # %% Data loaders
    x_train = train_noise + train_sigs
    y_train = np.stack([train_noise, train_sigs], axis=1)
    train_t = TensorDataset(torch.from_numpy(x_train[:, None, :].astype(np.complex64)), 
                            torch.from_numpy(y_train.astype(np.complex64)))
    train_loader = DataLoader(train_t, batch_size=batch_size, shuffle=True, drop_last=False)

    x_val = val_noise + val_sigs
    y_val = np.stack([val_noise, val_sigs], axis=1)
    val_t = TensorDataset(torch.from_numpy(x_val[:, None, :].astype(np.complex64)), 
                            torch.from_numpy(y_val.astype(np.complex64)))
    val_loader = DataLoader(val_t, batch_size=batch_size, shuffle=True, drop_last=False)

    x_test = test_noise + test_sigs
    y_test = np.stack([test_noise, test_sigs], axis=1)
    test_t = TensorDataset(torch.from_numpy(x_test[:, None, :].astype(np.complex64)), 
                            torch.from_numpy(y_test.astype(np.complex64)), torch.from_numpy(test_A[:, None]))
    test_loader = DataLoader(test_t, batch_size=batch_size, shuffle=False, drop_last=False)

    # %% Set up model and optimizer
    model = model_fn(win_len=args.win, win_inc=args.win_enc, rnn_layers=args.rnn_layers, rnn_units=args.rnn_units,
                     fft_len=args.fft_len, masking_mode=args.masking_mode, kernel_size=args.kernel_size, kernel_num=args.k_num,
                     use_clstm=True if args.use_clstm else False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # %% Train
    writer = SummaryWriter() 
    l = {'loss comparison': {'train/val': ['Multiline', ['loss/train', 'loss/validation']]}}
    writer.add_custom_scalars(l)
    
    ckpt = args.save_folder + os.path.sep + writer.log_dir.split(os.path.sep)[-1] + '.pt'
    best_loss = np.inf
    for epoch in tqdm(range(n_epochs)):
        batch_losses = []
        batch_mse = []
        for x, y in train_loader:
            model.train()
            x = x/args.scale
            x = torch.cat([x.real, x.imag], dim=-1)
            optimizer.zero_grad()
            _, yhat = model(x.to(device))
            yhat = yhat * args.scale
            yt = torch.cat([y[:, 1, :].real, y[:, 1, :].imag], dim=-1) 
            loss = loss_fn(yhat, yt.to(device)) # signal only
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            mse = losses_dict['complex_mse'](yhat, yt.to(device)) # signal only
            batch_mse.append(mse.item())
        print('Train loss %0.4f' % np.mean(np.array(batch_losses)))
        print('Train mse %0.4f' % np.mean(np.array(batch_mse)))
        tl_epoch = np.mean(np.array(batch_losses))
        writer.add_scalar('loss/train', tl_epoch, epoch)
        tmse_epoch = np.mean(np.array(batch_mse))
        writer.add_scalar('mse/train', tmse_epoch, epoch)
        batch_losses = []
        batch_mse = []
        for x, y in val_loader:
            model.eval()
            with torch.no_grad():
                x = x/args.scale
                x = torch.cat([x.real, x.imag], dim=-1)
                _, yhat = model(x.to(device))
                yhat = yhat*args.scale
                yt = torch.cat([y[:, 1, :].real, y[:, 1, :].imag], dim=-1) 
                loss = loss_fn(yhat, yt.to(device))
                batch_losses.append(loss.item())
                mse = losses_dict['complex_mse'](yhat, yt.to(device))
                batch_mse.append(mse.item())
        print('Val loss %0.4f' % np.mean(np.array(batch_losses)))
        print('Val mse %0.4f' % np.mean(np.array(batch_mse)))
        vl_epoch = np.mean(np.array(batch_losses))
        vmse_epoch = np.mean(np.array(batch_mse))
        writer.add_scalar('loss/validation', vl_epoch, epoch)
        writer.add_scalar('mse/validation', vmse_epoch, epoch)
        if vl_epoch < best_loss:
            print('Saving checkpoint...')
            state = {
                'net': model.state_dict(),
                'loss': vl_epoch,
                'epoch': epoch,
            }
            # Add args to output
            args_dict = vars(args)
            state.update(args_dict)
            torch.save(state, ckpt)
            best_loss = vl_epoch

    # %% Predict on test data
    noise_preds = []
    sig_preds = []
    batch_losses = []
    batch_mse = 0
    batch_idx = 0
    for x_test, y_test, A_test in test_loader:
        model.eval()
        with torch.no_grad():
            x = x_test / args.scale
            x = torch.cat([x.real, x.imag], dim=-1)
            _, yhat = model(x.to(device))
            yhat = yhat * args.scale
            yt = torch.cat([y_test[:, 1, :].real, y_test[:, 1, :].imag], dim=-1) 
            batch_losses.append(loss_fn(yhat, yt.to(device)).item())
            mse = losses_dict['complex_mse'](yhat/A_test.to(device), (yt/A_test).to(device))
            batch_mse += mse.item()
            yhat = torch.complex(yhat[:, :1024], yhat[:, 1024:])
            yhat = yhat.cpu().detach().numpy()
            noise_preds.append(np.nan*np.zeros_like(yhat))
            sig_preds.append(yhat)
            batch_idx += 1
    test_loss = np.mean(np.array(batch_losses))
    test_mse = batch_mse / batch_idx
    print('test mse% 0.3g' % test_mse)

    sig_preds = np.concatenate(sig_preds, 0).squeeze()
    noise_preds = np.concatenate(noise_preds, 0).squeeze()

    for i in range(np.min([10, sig_preds.shape[0]])):
        f = plt.figure(figsize=(8, 6))
        for j, (fn, lab) in enumerate(zip([np.real, np.imag], ['Real', 'Imag'])):
            plt.subplot(2,1,j+1)
            plt.plot(t, fn(test_sigs[i, :] + test_noise[i, :]), 'grey', label='Data', alpha=0.7)
            plt.plot(t, fn(test_sigs[i, :]), 'k', label='Signal')
            plt.plot(t, fn(sig_preds[i, :]), 'r', alpha=0.5, label='Reconstructed')
            plt.ylabel(lab)
            plt.legend()
        plt.tight_layout()
        writer.add_figure('Test predictions', f, i)
        plt.close()
        #plt.show()

    # hparams and test metrics
    hparams = {}
    hparams['model_DR'] = 'DCCRN'
    hparams['loss'] = args.loss_fn
    hparams['high_snr_data'] = args.high_snr_data
    hparams['central_f_zero'] = args.central_f_zero
    hparams['use_real_noise'] = args.use_real_noise
    writer.add_hparams(hparams, {'hparam/mse': test_mse, 'hparam/loss': test_loss}) 

    writer.flush()
    writer.close()

    # save model, metrics
    args_dict = vars(args)
    args_dict['test_mse'] = test_mse
    args_dict['test_loss'] = test_loss
    torch.save(args_dict, 'saved_models/%s_info.pt' % save_stub)
    torch.save(model.state_dict(), 'saved_models/%s_statedict.pt' % save_stub)


if __name__ == "__main__":

    global args

    parser = argparse.ArgumentParser(description='DCCRN training')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--scale', default=1.0, type=float, help='Rescale data divide by this')
    parser.add_argument('--n_epochs', default=50, type=int, help='Epochs')
    parser.add_argument('--save_folder',action='store', default='checkpoint', type=str, help='Folder to save trained model to')   
    parser.add_argument('--real_noise_path', action='store', default='/data/real_noise_file.pkl', help='full path/name of real noise file')
    parser.add_argument('--synth_data_path', action='store', default='/data/synth_data/', help='path to synthetic data')                 
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--use_real_noise', action='store_true', help='use real noise? (else white noise)')
    parser.add_argument('--high_snr_data', action='store_true', help='use high snr data? (else lower snr)')
    parser.add_argument('--central_f_zero', action='store_true', help='use central frequency around zero? (else oscillatory signal)')
    parser.add_argument('--use_cuda', action='store_true', help='enable cuda?')
    
    parser.add_argument('--check_finished', action='store_true', help='check if finished (if not rerun)?')
    
    # MOdel arguments
    parser.add_argument('--rnn_layers', default=2, type=int, help='rnn number of layers')
    parser.add_argument('--win', default=32, type=int, help='window length (similar to CTN...)')
    parser.add_argument('--rnn_units', default=128, type=int, help='rnn number units - i think it needs to match something or you might get size errors')
    parser.add_argument('--win_enc', default=16, type=int, help='similar to CTN stride which was by default half of win')
    parser.add_argument('--fft_len', default=512, type=int, help='fft len')
    parser.add_argument('--masking_mode', action='store',  default='E', type=str, help='masking mode, should be E, R, or C')
    parser.add_argument('--use_clstm', action='store_true', help='use clstm?')
    parser.add_argument('--kernel_size', default=5, type=int, help='kernel size')

    args = parser.parse_args()

    args.loss_fn = 'mse' 

    # Hard coded arguments can be changed if needed
    args.k_num = [16,32,64,128,256,256]
    args.win_enc = args.win // 2
    
    print(args)
    main()# %%
