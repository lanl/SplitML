import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from splitml.autoencoders import ComplexNet, DualRealNet
from splitml.activations import ComplexReLU
from splitml.losses import complex_mse


def model_complex(clean_training, noisy_training, clean_validation, noisy_validation, activation = ComplexReLU(), 
    loss_type = complex_mse, use_clean_training = True, use_clean_validation = True, learning_rate = 0.01, 
    patience = 5, epochs = 1000, seed = 1, time_points=1000):
    """
    Function for training and saving ComplexNet model
    """
    torch.manual_seed(seed)
    trigger_times = 0
    loss_training = []
    loss_validation = []
    device = torch.device('cpu') # set device

    # Get Data Ready for Training
    noisy_training_data = torch.tensor(noisy_training).to(device).type(torch.complex64)
    clean_training_data = torch.tensor(clean_training).to(device).type(torch.complex64)

    noisy_validation_data = torch.tensor(noisy_validation).to(device).type(torch.complex64)
    clean_validation_data = torch.tensor(clean_validation).to(device).type(torch.complex64)

    # Loss, Model and Optimizer 
    model = ComplexNet(activation = activation, t_input=time_points).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    # Train Network
    start_time = time.time()
    model.train() 
    for epoch in range(epochs):
        # Forward
        out = model(noisy_training_data)
        if use_clean_training is False:
            loss = loss_type(out, noisy_training_data)
        else:
            loss = loss_type(out, clean_training_data)
        # Backward
        optimizer.zero_grad() 
        loss.backward() 
        # Optimizer step to update weights
        optimizer.step()
        # Save training loss during each epoch
        loss_training.append(loss.item())
        # Save validation loss during each epoch
        with torch.no_grad(): 
            out = model(noisy_validation_data)
            if use_clean_validation is False:
                loss = loss_type(out, noisy_validation_data)
            else:
                loss = loss_type(out, clean_validation_data)
            loss_validation.append(loss.item())
        # Stop when validation loss goes up for more iterations than patience
        if epoch > 200:
            if loss_validation[-1] > loss_validation[-2]:
                trigger_times += 1
                if trigger_times >= patience:
                    # Stop training
                    break
            else:
                trigger_times = 0
    # Training time
    training_time = time.time()-start_time
    
    return model, loss_training, loss_validation, training_time

def model_dual_real1(clean_training, noisy_training, clean_validation, noisy_validation, activation = F.relu, 
    loss_type = nn.MSELoss(), use_clean_training = True, use_clean_validation = True, learning_rate = 0.01, 
    patience = 5, epochs = 1000, seed = 1, time_points=1000):
    """
    Function for training and saving DualReal1 model
    """
    torch.manual_seed(seed)
    trigger_times = 0
    loss_training = []
    loss_validation = []
    device = torch.device('cpu') # set device

    # Get Data Ready for Training
    noisy_training_data = torch.tensor(noisy_training).to(device).type(torch.complex64)
    clean_training_data = torch.tensor(clean_training).to(device).type(torch.complex64)

    noisy_validation_data = torch.tensor(noisy_validation).to(device).type(torch.complex64)
    clean_validation_data = torch.tensor(clean_validation).to(device).type(torch.complex64)

    # Loss, Model and Optimizer
    model = DualRealNet(activation = activation, t_input=time_points).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # Train Network
    start_time = time.time()
    model.train() 
    for epoch in range(epochs):
        # Forward for real component
        out_real = model(noisy_training_data.real)
        if use_clean_training is False:
            loss_real = loss_type(out_real, noisy_training_data.real)
        else:
            loss_real = loss_type(out_real, clean_training_data.real)
        # Backward
        optimizer.zero_grad() 
        loss_real.backward() 
        # Optimizer step to update weights
        optimizer.step()
        
        # Forward for imag component
        out_imag = model(noisy_training_data.imag)
        if use_clean_training is False:
            loss_imag = loss_type(out_imag, noisy_training_data.imag)
        else:
            loss_imag = loss_type(out_imag, clean_training_data.imag)
        # Backward
        optimizer.zero_grad()
        loss_imag.backward() 
        # Optimizer step to update weights
        optimizer.step()
        
        # Save training loss during each epoch
        loss_training.append(loss_real.item() + loss_imag.item())
        # Save validation loss during each epoch
        with torch.no_grad(): 
            out_real = model(noisy_validation_data.real)
            if use_clean_validation is False:
                loss_real = loss_type(out_real, noisy_validation_data.real)
            else:
                loss_real = loss_type(out_real, clean_validation_data.real)
            out_imag = model(noisy_validation_data.imag)
            if use_clean_validation is False:
                loss_imag = loss_type(out_imag, noisy_validation_data.imag)
            else:
                loss_imag = loss_type(out_imag, clean_validation_data.imag)
            loss_validation.append(loss_real.item() + loss_imag.item())
        # Stop when validation loss goes up for more iterations than patience
        if epoch > 200:
            if loss_validation[-1] > loss_validation[-2]:
                trigger_times += 1
                if trigger_times >= patience:
                    # Stop training
                    break
            else:
                trigger_times = 0
    # Training time
    training_time = time.time()-start_time  
    
    return model, loss_training, loss_validation, training_time

def model_dual_real2(clean_training, noisy_training, clean_validation, noisy_validation, 
    activation = F.relu, loss_type = nn.MSELoss(), use_clean_training = True, 
    use_clean_validation = True, learning_rate = 0.01, patience = 5, epochs = 1000, seed = 1, time_points=1000):
    """
    Function for training and saving DualReal2 model
    """
    torch.manual_seed(seed)
    trigger_times = 0
    loss_training = []
    loss_validation = []
    device = torch.device('cpu') # set device
    
    # Get Data Ready for Training
    noisy_training_data = torch.tensor(noisy_training).to(device).type(torch.complex64)
    clean_training_data = torch.tensor(clean_training).to(device).type(torch.complex64)

    noisy_validation_data = torch.tensor(noisy_validation).to(device).type(torch.complex64)
    clean_validation_data = torch.tensor(clean_validation).to(device).type(torch.complex64)

    # Loss, Model and Optimizer
    model_real = DualRealNet(activation = activation, t_input=time_points).to(device)
    model_imag = DualRealNet(activation = activation, t_input=time_points).to(device)
    optimizer_real = torch.optim.Adam(model_real.parameters(), lr=learning_rate, weight_decay=5e-4)
    optimizer_imag = torch.optim.Adam(model_imag.parameters(), lr=learning_rate, weight_decay=5e-4)

    # Train Networks
    start_time = time.time()
    model_real.train()
    model_imag.train() 
    for epoch in range(epochs):
        # Forward real network
        out_real = model_real(noisy_training_data.real)
        if use_clean_training is False:
            loss_real = loss_type(out_real, noisy_training_data.real)
        else:
            loss_real = loss_type(out_real, clean_training_data.real)
        # Backward
        optimizer_real.zero_grad()
        loss_real.backward()
        # Optimizer step to update weights 
        optimizer_real.step()
        
        # Forward imag network
        out_imag = model_imag(noisy_training_data.imag)
        if use_clean_training is False:
            loss_imag = loss_type(out_imag, noisy_training_data.imag)
        else:
            loss_imag = loss_type(out_imag, clean_training_data.imag)
        # Backward
        optimizer_imag.zero_grad()
        loss_imag.backward() 
        # Optimizer step to update weights        
        optimizer_imag.step()
        
        # Save training loss during each epoch
        loss_training.append(loss_real.item()+loss_imag.item())

        # Save validation loss during each epoch
        with torch.no_grad(): 
            out_real = model_real(noisy_validation_data.real)
            if use_clean_validation is False:
                loss_real = loss_type(out_real, noisy_validation_data.real)
            else:
                loss_real = loss_type(out_real, clean_validation_data.real)
            out_imag = model_imag(noisy_validation_data.imag)
            if use_clean_validation is False:
                loss_imag = loss_type(out_imag, noisy_validation_data.imag)
            else:
                loss_imag = loss_type(out_imag, clean_validation_data.imag)
            loss_validation.append(loss_real.item() + loss_imag.item())
        # Stop when validation loss goes up for more iterations than patience
        if epoch > 200:
            if loss_validation[-1] > loss_validation[-2]:
                trigger_times += 1
                if trigger_times >= patience:
                    # Stop training
                    break
            else:
                trigger_times = 0
    # Training time
    training_time = time.time()-start_time 

    return model_real, model_imag, loss_training, loss_validation, training_time

def model_dual_real1_concatenate(clean_training, noisy_training, clean_validation, noisy_validation, 
    activation = F.relu, loss_type = nn.MSELoss(), use_clean_training = True, 
    use_clean_validation = True, learning_rate = 0.01, patience = 5, epochs = 1000, seed = 1, time_points=1000):
    """
    Function for training and saving DualReal1 model
    """
    torch.manual_seed(seed)
    trigger_times = 0
    loss_training = []
    loss_validation = []
    device = torch.device('cpu') # set device

    # Get Data Ready for Training
    noisy_training_data = torch.tensor(noisy_training).to(device).type(torch.complex64)
    clean_training_data = torch.tensor(clean_training).to(device).type(torch.complex64)

    noisy_validation_data = torch.tensor(noisy_validation).to(device).type(torch.complex64)
    clean_validation_data = torch.tensor(clean_validation).to(device).type(torch.complex64)

    # Loss, Model and Optimizaer
    model = DualRealNet(activation = activation, t_input=time_points).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # Concatenate real and imaginary data into a real-valued 2-D time series
    noisy_training = torch.cat((noisy_training_data.real, noisy_training_data.imag),0)
    clean_training = torch.cat((clean_training_data.real, clean_training_data.imag),0)

    noisy_validation = torch.cat((noisy_validation_data.real, noisy_validation_data.imag),0)
    clean_validation = torch.cat((clean_validation_data.real, clean_validation_data.imag),0)

    # Train Network
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        # Forward
        out = model(noisy_training)
        if use_clean_training is False:
            loss = loss_type(out, noisy_training)
        else:
            loss = loss_type(out, clean_training)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Optimizer step to update weights
        optimizer.step()
        # Save training loss during each epoch
        loss_training.append(loss.item())
        # Save validation loss during each epoch
        with torch.no_grad():
            out = model(noisy_validation)
            if use_clean_validation is False:
                loss = loss_type(out, noisy_validation)
            else:
                loss = loss_type(out, clean_validation)
            loss_validation.append(loss.item())
        # Stop when validation loss goes up for more iterations than patience
        if epoch > 200:
            if loss_validation[-1] > loss_validation[-2]:
                trigger_times += 1
                if trigger_times >= patience:
                    # Stop training
                    break
            else:
                trigger_times = 0
    # Training time
    training_time = time.time()-start_time

    return model, loss_training, loss_validation, training_time