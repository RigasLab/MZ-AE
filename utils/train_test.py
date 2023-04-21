#%%
# import torch.nn as nn
# import numpy as np
# import scipy.io
import torch
import sys
sys.path.append('../')
from Lorenz_Datagen.L96_torch import L96TwoLevel_torch

#%%
def train_loss_bp(data_loader, model, loss_function, optimizer):
    '''
    Data_Loader -> X [batch_size seq_len state_dim]
    '''

    
    
    num_batches = len(data_loader)
    total_loss  = 0
    model.train()
    

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def test_loss(data_loader, model, loss_function):
    '''
    Data_Loader -> X [batch_size seq_len state_dim]
    '''
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            # print("output shape: ", output.shape)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    return avg_loss

def predict(data_loader, model, device = 'cpu'):
    '''
    Data_Loader -> X [batch_size seq_len state_dim]
    '''
    output = torch.tensor([]).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output

def coupled_train_model(data_loader, model, loss_function, optimizer, data_args, device):
    '''
    train the model coupled with l96 solver
    Data_Loader -> X [batch_size seq_len state_dim]
    '''
    num_batches = len(data_loader)
    total_loss  = 0
    model.train()

    for X, y in data_loader:
        zn1 = model(X)

        #calculating for X at next time step
        l96 = L96TwoLevel_torch(X_init = X[:,-1,:], Y_init = None, dt = data_args["dt"], save_dt = data_args["save_dt"], noYhist = True, parameterization = True, device = device)
        l96.step(Bdata = zn1)
        X1 = torch.squeeze(l96.X.clone())
        loss = loss_function(X1, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

def coupled_test_model(data_loader, model, loss_function, data_args, device):
    '''
    test the model coupled with l96 sover
    Data_Loader -> X [batch_size seq_len state_dim]
    '''
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            zn1 = model(X)
            #calculating for X at next time step
            l96 = L96TwoLevel_torch(X_init = X[:,-1,:], Y_init = None, dt = data_args["dt"], save_dt = data_args["save_dt"], noYhist = True, parameterization = True, device = device)
            l96.step(Bdata = zn1)
            X1 = torch.squeeze(l96.X.clone())
            loss = loss_function(X1, y)
            # print("output shape: ", output.shape)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss



def coupled_train_model2(data_loader, model, loss_function, optimizer, data_args, device):
    '''
    train the model coupled with l96 solver
    Data_Loader -> X [batch_size seq_len state_dim]
    '''
    num_batches = len(data_loader)
    total_loss  = 0
    model.train()

    for X, X1_target, zn1_target in data_loader:
        zn1 = model(X)

        #calculating for X at next time step
        l96 = L96TwoLevel_torch(X_init = X[:,-1,:], Y_init = None, dt = data_args["dt"], save_dt = data_args["save_dt"], noYhist = True, parameterization = True, device = device)
        l96.step(Bdata = zn1)
        X1 = torch.squeeze(l96.X.clone())
        loss1 = loss_function(X1, X1_target)
        loss2 = loss_function(zn1, zn1_target)
        loss = 100*loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss, loss1, loss2

def coupled_test_model2(data_loader, model, loss_function, data_args, device):
    '''
    test the model coupled with l96 sover
    Data_Loader -> X [batch_size seq_len state_dim]
    '''
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, X1_target, zn1_target in data_loader:
            zn1 = model(X)
            #calculating for X at next time step
            l96 = L96TwoLevel_torch(X_init = X[:,-1,:], Y_init = None, dt = data_args["dt"], save_dt = data_args["save_dt"], noYhist = True, parameterization = True, device = device)
            l96.step(Bdata = zn1)
            X1 = torch.squeeze(l96.X.clone())
            loss1 = loss_function(X1, X1_target)
            loss2 = loss_function(zn1, zn1_target)
            loss  = 100*loss1 + loss2
            # print("output shape: ", output.shape)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss, loss1, loss2