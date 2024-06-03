import numpy as np
from sklearn.preprocessing import StandardScaler
# Model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
# CV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
# Numpy
from numpy import mean
from numpy import std
from func import ml_cv_plot
from joblib import dump, load

CV = 0
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
if CV:
    fold = 5
    # Define model
    model_list = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), GaussianProcessRegressor()]
    result_avg = []
    result_std = []
    for i in model_list:
        model = i
        cv = RepeatedKFold(n_splits=fold, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, x_array, y_array.ravel(), scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        result_avg.append(mean(n_scores))
        result_std.append(std(n_scores))
    index = np.arange(len(model_list))
    MAE = -np.array(result_avg)
    # CV plot
    ml_cv_plot(index, MAE, result_std)
best_model = "GP"  # , "LR", KNN, DT, RF
if best_model == "GP":
    model = GaussianProcessRegressor()
if best_model == "LR":
    model = LinearRegression()
if best_model == "KNN":
    model = KNeighborsRegressor()
if best_model == "DT":
    model = DecisionTreeRegressor()
if best_model == "RF":
    model = RandomForestRegressor()
model.fit(x_array, y_array.ravel())
# save
dump(model, best_model + "_model.pkl", compress=1)
#