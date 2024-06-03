from model import get_model, Optimization
from func import pred_plot, pred_plot_compare, output_full_profile
import glob
import os
import time
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "ann"

# Define A-coil & flow rate
Acoil = 1  # 1: A-coil 24C-B, 2: A-coil 36C-B, 3: A-coil 36C-C, 4: A-coil 48C-C
FR = 1200  # flow rate: 470-1600 [CFM]

# load A-coil geometric info.
path = r'./data' # relative path
all_files = glob.glob(os.path.join(path + '\LGE_Conds_Input', "A-coil" + str(Acoil) + "*u_" + str(654) + "*.csv")) # input template file
dg = pd.concat(map(pd.read_csv, all_files), ignore_index=True)
dg['CFM'] = FR

# import actual/closest cases for comparing
plot_reference_profile = 0
compare_w_existing_actual_profile = 0
FR_test = 1200
if compare_w_existing_actual_profile:
    # Compare to actual out-of-sample cases
    dh = pd.read_csv(path + '\OS_200' + "\A-coil" + str(Acoil) + '_u_' + str(FR_test) + '_SR_200.csv')
else:
    # Compare to exisiting closest cases
    dh = pd.read_csv(path + '\DS_200_half' + "\A-coil" + str(Acoil) + '_u_' + str(FR_test) + '_SR_200.csv')

model_input = dg
train_dataset_s = model_input
train_features_s = train_dataset_s.copy()
train_labels_s = train_features_s.pop('Air Velocity [m/s]')
x_array = train_features_s.to_numpy()
y_array = train_labels_s.to_numpy()

# load normalization results from training
std_scaler_s = load('std_scaler.bin')
test_features_scaled_s = std_scaler_s.transform(x_array)
x_test = test_features_scaled_s.astype('float64')
y_test = y_array.astype('float64')
test_features = torch.Tensor(x_test)
test_targets = torch.Tensor(y_test)
test = TensorDataset(test_features, test_targets)

# Data Loader
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
# Model setting
hidden_dim = 2**8
layer_dim = 2
input_dim = 4
output_dim = 1
model_params = {'input_dim': input_dim,
                'hidden_dim': round(hidden_dim),
                'layer_dim': round(layer_dim),
                'output_dim': output_dim}
model = get_model(model_name, model_params)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")
# Load trained model
if device == "cuda:0":
    model.load_state_dict(torch.load("Acoil_ann.pth"))  # using gpu
else:
    model.load_state_dict(torch.load("Acoil_ann.pth", map_location='cpu'))  # using cpu
model.to(device)
model.eval()

# Optimizer setting
learning_rate = 1e-3
weight_decay = 1e-6
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.MSELoss(reduction="mean")
opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
start = time.time()
print("Prediction start")

# Predicting
predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)
end = time.time()
print("Elapsed time:", end - start)
preds = np.asarray(predictions).reshape(-1, 1)
# vals = np.asarray(values).reshape(1,-1)

# Results plot
if plot_reference_profile:
    pred_plot_compare(model_input, preds, dh)
else:
    pred_plot(model_input, preds)
#
# Filled the Air velocity
model_input['Air Velocity [m/s]'] = preds.reshape(-1, 1)

# Output full profile data
output_path = path # + '\LGE_Conds_Output'
df_out = output_full_profile(model_input)
SR = len(df_out)
df_out.to_csv(output_path+'\A-coil' + str(Acoil) + '_u_' + str(FR) + 'CFM_SR_' + str(SR) + '_even_full.csv', index=False)