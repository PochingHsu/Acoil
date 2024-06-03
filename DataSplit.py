import glob
import pandas as pd
import os
import numpy as np
import random

# Loading CFD data
DS = 200  # sample rate
FR = [470, 510, 550, 590, 630, 670, 700, 710, 750, 760, 790, 820, 830, 870, 880, 910,
                   940, 950, 990, 1000, 1030, 1060, 1070, 1100, 1120, 1180, 1200, 1240, 1300, 1360,
                   1400, 1420, 1480, 1540, 1600]  # CFM
split_ratio = 0.7
# train/val data split based on FR
train_FR = random.sample(FR, round(len(FR)*split_ratio))
test_FR = list(set(FR) - set(train_FR))
path = r'D:\LGE\Air coil\IDU\CFD_profile\DS_' + str(DS) + '_half'  # modify accordingly
all_files = glob.glob(os.path.join(path, "A-coil"+"*u*.csv"))
df = pd.concat(map(pd.read_csv, all_files), ignore_index=True)

train_features_s = df[df['CFM'].isin(train_FR)]
train_labels_s = train_features_s.pop('Air Velocity [m/s]')
x_train = train_features_s.to_numpy()
y_train = train_labels_s.to_numpy()

test_features_s = df[df['CFM'].isin(test_FR)]
test_labels_s = test_features_s.pop('Air Velocity [m/s]')
x_test = test_features_s.to_numpy()
y_test = test_labels_s.to_numpy()
# save to csv file
np.save('x_train', x_train)
np.save('y_train', y_train)
np.save('x_test', x_test)
np.save('y_test', y_test)
