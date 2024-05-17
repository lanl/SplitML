import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from splitml.transformers import ComplexTransformer, DRTransformer, DRCTransformer
from splitml.activations import ComplexReLU
from splitml.losses import complex_mse

def train_dr(noisy_training, clean_training, noisy_validation, clean_validation,
             d_model=256, num_heads=4, num_layers=2, d_ff=256, dropout=0.1,
             loss_type=nn.MSELoss(), seed=1, epochs=100, device = torch.device('cpu')):

    max_seq_length = len(noisy_training[0])
    src_data = torch.from_numpy(np.concatenate((noisy_training.real,noisy_training.imag),axis=0)).float().unsqueeze(-1)
    tgt_data = torch.from_numpy(np.concatenate((clean_training.real,clean_training.imag),axis=0)).float().unsqueeze(-1)

    src_data_validation = torch.from_numpy(np.concatenate((noisy_validation.real,noisy_validation.imag),axis=0)).float().unsqueeze(-1)
    tgt_data_validation = torch.from_numpy(np.concatenate((clean_validation.real,clean_validation.imag),axis=0)).float().unsqueeze(-1)

    np.random.seed(seed)
    torch.manual_seed(seed)
    loss_training_save = []
    loss_validation_save = []

    # Train Network
    start_time = time.time()
    
    DRtransformer = DRTransformer(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)

    criterion = loss_type
    optimizer = optim.Adam(DRtransformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    DRtransformer.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = DRtransformer(src_data.to(device))
        loss_training = criterion(output.contiguous().view(-1, 1), tgt_data.contiguous().view(-1, 1).to(device)) 
        loss_training_save.append(loss_training.item())
        loss_training.backward()
        optimizer.step()

        with torch.no_grad():
            DRtransformer.eval()
            output_validation = DRtransformer(src_data_validation.to(device))

            loss_validation = criterion(output_validation.contiguous().view(-1, 1), tgt_data_validation.contiguous().view(-1, 1).to(device)) 
            loss_validation_save.append(loss_validation.item())
            DRtransformer.train()

    training_time = time.time()-start_time

    return DRtransformer, loss_training, loss_validation, training_time



def train_drc(noisy_training, clean_training, noisy_validation, clean_validation,
              d_model=256, num_heads=4, num_layers=2, d_ff=256, dropout=0.1,
             loss_type=nn.MSELoss(),  seed=1, epochs=100, device = torch.device('cpu')):

    max_seq_length = len(noisy_training[0])

    src_data = torch.from_numpy(np.stack((noisy_training.real,noisy_training.imag), axis=2)).float()
    tgt_data = torch.from_numpy(np.stack((clean_training.real,clean_training.imag), axis=2)).float()

    src_data_validation = torch.from_numpy(np.stack((noisy_validation.real,noisy_validation.imag), axis=2)).float()
    tgt_data_validation = torch.from_numpy(np.stack((clean_validation.real,clean_validation.imag), axis=2)).float()

    np.random.seed(seed)
    torch.manual_seed(seed)
    loss_training_save = []
    loss_validation_save = []

    # Train Network
    start_time = time.time()

    DRCtransformer = DRCTransformer(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)

    criterion = loss_type
    optimizer = optim.Adam(DRCtransformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    DRCtransformer.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = DRCtransformer(src_data.to(device))
        loss_training = criterion(output.contiguous().view(-1, 2), tgt_data.contiguous().view(-1, 2).to(device)) 
        loss_training_save.append(loss_training.item()) 
        loss_training.backward()
        optimizer.step()

        with torch.no_grad():
            DRCtransformer.eval()
            output_validation = DRCtransformer(src_data_validation.to(device))

            loss_validation = criterion(output_validation.contiguous().view(-1, 2), tgt_data_validation.contiguous().view(-1, 2).to(device)) 
            loss_validation_save.append(loss_validation.item())
            DRCtransformer.train()

    training_time = time.time()-start_time

    return DRCtransformer, loss_training, loss_validation, training_time
      


def train_cn(noisy_training, clean_training, noisy_validation, clean_validation,
             d_model=256, num_heads=4, num_layers=2, d_ff=256, dropout=0.1,
             loss_type=complex_mse, seed=1, epochs=100, device = torch.device('cpu')):

    max_seq_length = len(noisy_training[0])

    # Set data
    src_data = torch.from_numpy(noisy_training).cfloat().unsqueeze(-1)
    tgt_data = torch.from_numpy(clean_training).cfloat().unsqueeze(-1)

    src_data_validation = torch.from_numpy(noisy_validation).cfloat().unsqueeze(-1)
    tgt_data_validation = torch.from_numpy(clean_validation).cfloat().unsqueeze(-1)

    np.random.seed(seed)
    torch.manual_seed(seed)
    loss_training_save = []
    loss_validation_save = []

    # Train Network
    start_time = time.time()

    Complextransformer = ComplexTransformer(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)

    criterion = loss_type 
    optimizer = optim.Adam(Complextransformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    Complextransformer.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = Complextransformer(src_data.to(device))
        loss_training = criterion(output.contiguous().view(-1, 1), tgt_data.contiguous().view(-1, 1).to(device)) 
        loss_training_save.append(loss_training.item())
        loss_training.backward()
        optimizer.step()

        with torch.no_grad():
            Complextransformer.eval()

            output_validation = Complextransformer(src_data_validation.to(device))
            loss_validation = criterion(output_validation.contiguous().view(-1, 1), tgt_data_validation.contiguous().view(-1, 1).to(device))
            loss_validation_save.append(loss_validation.item()/2)

            Complextransformer.train()

    training_time = time.time()-start_time
    return Complextransformer, loss_training, loss_validation, training_time
        
