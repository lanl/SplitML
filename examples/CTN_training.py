"""
Script to fit ConvTasNet on synthetic NQR data.

Author: Natalie Klein
"""
import os
from inspect import getmembers, isfunction, isclass
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from splitml import losses
from splitml import activations
from splitml.data import synthetic_data_gen
from splitml.dualrealconvtasnet import DualRealConvTasNet
from splitml.dualrealchannelsconvtasnet import DualRealChannelsConvTasNet
from splitml.complexconvtasnet import ComplexConvTasNet

losses_dict = {k: v for (k, v) in getmembers(losses, isfunction)}
losses_dict['mse'] = torch.nn.MSELoss()
activs_dict = {k: v for (k, v) in getmembers(activations, isclass)}
activs_dict['PReLU'] = torch.nn.PReLU
activs_dict['Sigmoid'] = torch.nn.Sigmoid
models_dict = {'DualRealConvTasNet': DualRealConvTasNet,
                'DualReal2ConvTasNet': DualRealConvTasNet, 
                'DualRealChannelsConvTasNet': DualRealChannelsConvTasNet, 
                'ComplexConvTasNet': ComplexConvTasNet}

def main():

    if args.use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)

    # %% Settings 
    model_fn = models_dict[args.model]
    seed = args.seed
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.n_epochs
    loss_fn = losses_dict[args.loss_fn]
    # model kwargs
    m_kwargs = {}
    m_kwargs['enc_kernel_size'] = args.win
    m_kwargs['enc_num_feats'] = args.enc_dim
    m_kwargs['enc_depth'] = args.enc_depth
    m_kwargs['enc_activate'] = activs_dict[args.activ]
    m_kwargs['stft_layer'] = args.stft_layer
    m_kwargs['msk_kernel_size'] = args.msk_kern_size
    m_kwargs['msk_num_feats'] = args.msk_num_feats
    m_kwargs['msk_num_hidden_feats'] = args.msk_num_hidden_feats
    m_kwargs['msk_num_layers'] = args.msk_num_layers
    m_kwargs['msk_num_stacks'] = args.msk_num_stacks
    m_kwargs['dc_activ'] = activs_dict[args.dc_activ]
    m_kwargs['msk_activate'] = activs_dict[args.msk_activ]
    if args.share_encoder is not None:
        m_kwargs['share_encoder'] = args.share_encoder
    if args.share_decoder is not None:
        m_kwargs['share_decoder'] = args.share_decoder
    if args.share_mask is not None:
        m_kwargs['share_mask'] = args.share_mask
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    nt = 1024
    t = np.linspace(0, 0.018414, nt)

    # Check if alredy completed
    snr = 'high' if args.high_snr_data else 'low'
    fc = 'f0' if args.central_f_zero else 'f1600'
    noisedist = 'whitenoise'
    save_stub = '%s_win%d_seed%d_%sSNR_%s_%s_%s' % (args.model, args.win, args.seed, snr, fc, noisedist, args.loss_fn)
    if os.path.isfile('saved_models/%s_info.pt' % save_stub):
        print('already completed')
        return

    # %% generate data
    (train_sigs, train_noisy, val_sigs, val_noisy, 
        test_sigs, test_noisy, sig_params_training, sig_params_validation, 
        sig_params_testing, t) = synthetic_data_gen(N = 10000, nt = 1024, fs = 1./1.8e-05, 
                                                    w_range = [-50,50], phi_range = [-np.pi, np.pi], T2_range = [1e-3, 1e-2], 
                                                    sigma_range = [1e-3, 1e-2], A_range = [1,2], s = 1)

    train_noise = train_noisy - train_sigs
    val_noise = val_noisy - val_sigs
    test_noise = test_noisy - test_sigs

    test_A = sig_params_testing['A'].values[:, None]

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
                            torch.from_numpy(y_test.astype(np.complex64)), torch.from_numpy(test_A))
    test_loader = DataLoader(test_t, batch_size=batch_size, shuffle=False, drop_last=False)

    # %% Set up model and optimizer
    model = model_fn(**m_kwargs).to(device)
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
            optimizer.zero_grad()
            yhat = model(x.to(device))
            loss = loss_fn(yhat, y.to(device)) 
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            mse = losses_dict['complex_mse'](yhat[:, 1, :], y[:, 1, :].to(device))
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
                yhat = model(x.to(device))
                loss = loss_fn(yhat, y.to(device))
                batch_losses.append(loss.item())
                mse = losses_dict['complex_mse'](yhat[:, 1, :], y[:, 1, :].to(device))
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
            yhat = model(x_test.to(device))
            batch_losses.append(loss_fn(yhat, y_test.to(device)).item())
            mse = losses_dict['complex_mse'](yhat[:, 1, :]/A_test.to(device), (y_test[:, 1, :]/A_test).to(device))
            batch_mse += mse.item()
            yhat = yhat.cpu().detach().numpy()
            noise_preds.append(yhat[:, 0, :])
            sig_preds.append(yhat[:, 1, :])
            batch_idx += 1
    test_loss = np.mean(np.array(batch_losses))
    test_mse = batch_mse / batch_idx
    print('test mse% 0.3g' % test_mse)

    sig_preds = np.concatenate(sig_preds, 0)
    noise_preds = np.concatenate(noise_preds, 0)

    for i in range(np.min([10, sig_preds.shape[0]])):
        f = plt.figure(figsize=(8, 6))
        for j, (fn, lab) in enumerate(zip([np.real, np.imag], ['Real', 'Imag'])):
            plt.subplot(2,1,j+1)
            plt.plot(t, fn(test_sigs[i, :] + test_noise[i, :]), 'grey', label='Data', alpha=0.7)
            plt.plot(t, fn(test_sigs[i, :]), 'k', label='Signal')
            plt.plot(t, fn(sig_preds[i, :]), 'r', alpha=0.5, label='Reconstructed')
            #plt.suptitle('SNR %0.2f dB' % df.loc[df.set=='test', 'snr'].iloc[i])
            plt.ylabel(lab)
            plt.legend()
        plt.tight_layout()
        writer.add_figure('Test predictions', f, i)
        plt.close()
        #plt.show()

    # hparams and test metrics
    hparams = {}
    for k in m_kwargs.keys():
        if isclass(m_kwargs[k]):
            hparams[k] = m_kwargs[k].__name__
        else:
            hparams[k] = m_kwargs[k]
    hparams['model'] = args.model
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

    parser = argparse.ArgumentParser(description='ConvTasNet training')
    parser.add_argument('--model',action='store', default='DualRealChannelsConvTasNet', type=str, help='Model name')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--n_epochs', default=50, type=int, help='Epochs')
    parser.add_argument('--save_folder',action='store', default='checkpoint', type=str, help='Folder to save trained model to')                    
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--win', default=128, type=int, help='encoder kernel size')
    parser.add_argument('--enc_dim', default=512, type=int, help='encoder number of features')
    parser.add_argument('--enc_depth', default=3, type=int, help='encoder number of layers')
    parser.add_argument('--stft_layer', action='store_true', help='STFT layer?')
    parser.add_argument('--msk_kern_size', default=3, type=int, help='mask kernel size')
    parser.add_argument('--msk_num_feats', default=128, type=int, help='mask number of features')
    parser.add_argument('--msk_num_hidden_feats', default=16, type=int, help='mask number of hidden features')
    parser.add_argument('--msk_num_layers', default=8, type=int, help='mask number of layers')
    parser.add_argument('--msk_num_stacks', default=3, type=int, help='mask num stacks')
    parser.add_argument('--high_snr_data', action='store_true', help='use high snr data? (else lower snr)')
    parser.add_argument('--central_f_zero', action='store_true', help='use central frequency around zero? (else oscillatory signal)')
    parser.add_argument('--use_cuda', action='store_true', help='enable cuda?')

    args = parser.parse_args()

    # args where default depends on model - not available as command line args
    if args.model == 'ComplexConvTasNet':
        args.loss_fn = 'complex_logmse'
        args.activ = 'ComplexPReLU'
        args.dc_activ = 'ComplexPReLU'
        args.msk_activ = 'ComplexSigmoid'
        args.share_encoder = None
        args.share_decoder = None
        args.share_mask = None
    elif args.model == 'DualRealConvTasNet':
        args.loss_fn = 'complex_logmse' 
        args.activ = 'PReLU'
        args.dc_activ = 'PReLU'
        args.msk_activ = 'Sigmoid'
        args.share_encoder = True 
        args.share_decoder = True
        args.share_mask = True
    elif args.model == 'DualReal2ConvTasNet':
        args.loss_fn = 'complex_logmse' 
        args.activ = 'PReLU'
        args.dc_activ = 'PReLU'
        args.msk_activ = 'Sigmoid'
        args.share_encoder = False
        args.share_decoder = False
        args.share_mask = False
    elif args.model == 'DualRealChannelsConvTasNet':
        args.loss_fn = 'complex_logmse' 
        args.activ = 'PReLU'
        args.dc_activ = 'PReLU'
        args.msk_activ = 'Sigmoid'
        args.share_encoder = True 
        args.share_decoder = True
        args.share_mask = True
    else:
        raise ValueError('invalid model choice')
    
    print(args)
    main()
