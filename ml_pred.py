import glob
import os
import numpy as np
import pandas as pd
from joblib import dump, load
# Model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from func import pred_plot,pred_plot_compare, output_full_profile

# Define A-coil & flow rate
Acoil = 1  # 1: A-coil 24C-B, 2: A-coil 36C-B, 3: A-coil 36C-C, 4: A-coil 48C-C
FR = 300  # flow rate: <470, >1600 [CFM]
num_tube = 54  # the number of tube defined in CoilDesigner
#
# load A-coil geometric info.
path = r'./data' # relative path
all_files = glob.glob(os.path.join(path + '\LGE_Conds_Input', "A-coil" + str(Acoil) + "*u_" + str(654) + "*.csv"))
dg = pd.concat(map(pd.read_csv, all_files), ignore_index=True)
dg['CFM'] = FR

# import actual/closet cases for comparing
plot_reference_profile = 0
compare_w_existing_actual_profile = 1
FR_test = 300
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
x_test = std_scaler_s.transform(x_array)

# load trained model
model_name = glob.glob(os.path.join("*_model.pkl"))[0].split('_')[0]
print(model_name)
model = load(model_name + "_model.pkl")

if model_name == "GP":
    preds, preds_std = model.predict(x_test, return_std=True)
else:
    preds = model.predict(x_test)

# Results plot
if plot_reference_profile:
    pred_plot_compare(model_input, preds, dh)
else:
    pred_plot(model_input, preds)

# Filled the Air velocity
model_input['Air Velocity [m/s]'] = preds.reshape(-1, 1)

# Output full profile data
output_path = path # + '\LGE_Conds_Output'
df_out = output_full_profile(model_input)
SR = len(df_out)
df_out = df_out['Air Velocity [m/s]'].to_numpy()
df_out_transposed = np.transpose(df_out)
df_CD = np.tile(df_out_transposed, (num_tube, 1))
df_CD = pd.DataFrame(df_CD)
#df_out.to_csv(output_path+'\A-coil' + str(Acoil) + '_u_' + str(FR) + 'CFM_SR_' + str(SR) + '_even_full.csv', index=False)
df_CD.to_csv(output_path+'\A-coil' + str(Acoil) + '_u_' + str(FR) + 'CFM_SR_' + str(SR) + '_full_matrix.csv', header=False, index=False)