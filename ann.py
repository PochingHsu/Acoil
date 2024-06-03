import glob
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import get_model, Optimization, HyperparametersOP
from func import unpack
from hyperopt import hp, fmin, tpe, atpe, Trials
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "ann"
HP_tuning = 0  # Hyperparameter tuning
HPOP = "BO"  # Hyperparameter tuning method

# # Loading CFD data
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# Normalize the data
std_scaler_s = StandardScaler()
x_array = np.concatenate((x_train, x_test), axis=0)
y_array = np.concatenate((y_train, y_test), axis=0)
x_array = std_scaler_s.fit_transform(x_array)
x_train = std_scaler_s.transform(x_train)
x_test = std_scaler_s.transform(x_test)
dump(std_scaler_s, 'std_scaler.bin', compress=True)

x_array = x_array.astype('float64')
y_array = y_array.astype('float64')
x_array = torch.Tensor(x_array)
y_array = torch.Tensor(y_array)
dataset = TensorDataset(x_array, y_array)

x_train = x_train.astype('float64')
y_train = y_train.astype('float64')
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = x_test.astype('float64')
y_test = y_test.astype('float64')
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)
train = TensorDataset(x_train, y_train)
test = TensorDataset(x_test, y_test)
if HP_tuning:
    input_dim = x_array.shape[1]
    output_dim = y_array.shape[1]
    k_folds = 5
    n_epochs = 1000
    hyperop = HyperparametersOP(input_dim, output_dim, dataset, k_folds, n_epochs, device)
    if model_name == "ann":
        model = hyperop.ann_bo
    if HPOP == "BO":
        # hyperparameters searching space
        num_hidden_units_space = hp.quniform('num_hidden_units', 4, 10, 1)
        layer_dim_space = hp.quniform('layer_dim', 1, 2, 1)
        batch_size_power_space = hp.quniform('batch_size_power', 4, 10, 1)
        params_nn = {'batch_size_power': batch_size_power_space, 'num_hidden_units': num_hidden_units_space,
                     'layer_dim': layer_dim_space}
        tpe_trials = Trials()
        ann_best_param = fmin(fn=model,
                              space=params_nn,
                              max_evals=15,
                              rstate=np.random.default_rng(42),
                              algo=tpe.suggest,
                              trials=tpe_trials)
        trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in tpe_trials])
        trials_df["loss"] = [t["result"]["loss"] for t in tpe_trials]
        trials_df["trial_number"] = trials_df.index
        print(trials_df)
        pd.DataFrame(trials_df).to_csv(HPOP + '_' + model_name + "_A_coil" + ".csv", index=False)

else:

    # Data Loader
    batch_size = 2**6
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True)
    # Model setting
    hidden_dim = 2**8
    layer_dim = 2
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    model_params = {'input_dim': input_dim,
                    'hidden_dim': round(hidden_dim),
                    'layer_dim': round(layer_dim),
                    'output_dim': output_dim}
    model = get_model(model_name, model_params)
    model = model.to(device)
    # Optimizer setting
    learning_rate = 1e-3
    weight_decay = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss(reduction="mean")
    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    start = time.time()
    print("Training start")
    # Training
    n_epochs = 1000
    opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
    opt.plot_losses()
    end = time.time()
    print("Elapsed time:", end - start)
    # Save model
    torch.save(model.state_dict(), "Acoil_ann.pth")
