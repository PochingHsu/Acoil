import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import glob
import os
#

def CFD_downsample():
    Acoil = 4
    SR = 200  # sample rate
    path = r'D:\LGE\Air coil\IDU\CFD_profile\Acoil' + str(Acoil) + '_u_' + 'fixed'
    all_files = glob.glob(os.path.join(path, "A-coil" + str(Acoil) + "*u*.csv"))
    df = pd.concat(map(pd.read_csv, all_files), ignore_index=True)
    FR = np.array([470, 510, 550, 590, 630, 670, 700, 710, 750, 760, 790, 820, 830, 870, 880, 910,
                      940, 950, 990, 1000, 1030, 1060, 1070, 1100, 1120, 1180, 1200, 1240, 1300, 1360,
                     1400, 1420, 1480, 1540, 1600]) # CFM
    for i in FR:
        df2 = df[df['CFM'] == i]
        DS_ratio = len(df2) / SR
        df_DS = df2.groupby(np.arange(len(df2)) // DS_ratio).mean()
        x_half = df_DS['y-coordinate'].to_numpy().reshape(-1, 1)
        y_half = df_DS['x-velocity'].to_numpy().reshape(-1, 1)
        y = y_half
        x = x_half
        x = x / max(x)  # normalize
        df_x = pd.DataFrame(x, columns=['Normalized Coil Height [-]'])
        df_y = pd.DataFrame(y, columns=['Air Velocity [m/s]'])
        df_out = pd.concat([df_x, df_y], axis=1)
        # outout downsample csv
        df_DS_x = pd.DataFrame(df_out, columns=['Normalized Coil Height [-]'])
        df_DS_y = pd.DataFrame(df_out, columns=['Air Velocity [m/s]'])
        df_DS_out = pd.concat([df_DS_x, df_DS_y], axis=1)
        df_DS_out['CFM'] = i
        if Acoil == 1:
            df_DS_out['angle'] = 17
            df_DS_out['height'] = 0.38
        if Acoil == 2:
            df_DS_out['angle'] = 23.4
            df_DS_out['height'] = 0.53
        if Acoil == 3:
            df_DS_out['angle'] = 17.8
            df_DS_out['height'] = 0.53
        if Acoil == 4:
            df_DS_out['angle'] = 13.6
            df_DS_out['height'] = 0.63
        SR = len(df_DS_out)
        df_DS_out.to_csv("A-coil"+ str(Acoil)+'_u_' + str(i) +'_SR_'+ str(SR)+'test.csv', index=False)


def res_csv(acoil, mesh):
    file_list = glob.glob("hx_inlet_res_*"+mesh)
    for j in file_list:
        name_split = j.split("_")
        CFM = name_split[-3]
        Mesh = name_split[-2]
        df = pd.read_csv(j, delimiter="\t")
        df.reset_index(inplace=True)
        b = df[df['index'] == '((xy/key/label "x-velocity")'].index[0]
        c = df[df['index'] == '((xy/key/label "y-velocity")'].index[0]
        d = df[df['index'] == '((xy/key/label "k")'].index[0]
        e = df[df['index'] == '((xy/key/label "epsilon")'].index[0]
        # e = df[df['index'] == '((xy/key/label "omega")'].index[0]
        con = df.loc[0:b - 2].dropna()['((xy/key/label "continuity")'].reset_index(inplace=None)
        x_vel = df.loc[b + 1:c - 2].dropna()['((xy/key/label "continuity")'].reset_index(inplace=None)
        y_vel = df.loc[c + 1:d - 2].dropna()['((xy/key/label "continuity")'].reset_index(inplace=None)
        k = df.loc[d + 1:e - 2].dropna()['((xy/key/label "continuity")'].reset_index(inplace=None)
        eps = df.loc[e + 1:e + 1 + len(x_vel)].dropna()['((xy/key/label "continuity")'].reset_index(inplace=None)
        dg = pd.concat([con, x_vel, y_vel, k, eps], axis=1)
        dg = dg.drop(['index'], axis=1)
        dg.columns = ['continuity', 'x-velocity', 'y-velocity', 'k', 'epsilon']
        dg['CFM'] = int(CFM)
        dg['Mesh'] = Mesh
        dg = dg.tail(1)
        name = acoil + '_res_' + CFM + '_' + Mesh + '.csv'
        dg.to_csv(name, index=False)
        return Mesh


def res_summary(acoil, mesh, path):
    all_files = glob.glob(os.path.join(path, acoil + "*res*" + mesh + "*.csv"))
    df_res = pd.concat(map(pd.read_csv, all_files), ignore_index=True)
    df_res = df_res.sort_values('CFM').reset_index(drop=True)
    df_res.to_csv(acoil + '_res_' + mesh + '.csv', index=False)


def u_csv(acoil, mesh):
    file_list = glob.glob("hx_inlet_u_*"+mesh)
    for j in file_list:
        name_split = j.split("_")
        CFM = name_split[-3]
        Mesh = name_split[-2]
        df = pd.read_csv(j, sep=",", index_col=0)
        df.reset_index(drop=True, inplace=True)
        df = df.rename(columns=lambda x: x.strip())
        df['CFM'] = int(CFM)
        df['Mesh'] = Mesh
        name = acoil + '_u_' + CFM + '_' + Mesh + '.csv'
        df.to_csv(name, index=False)
        return Mesh

def max_vel_summary(acoil, path):
    all_files = glob.glob(os.path.join(path, acoil+"*_u_*.csv"))
    df_merge = pd.concat(map(pd.read_csv, all_files), ignore_index=True)
    df_merge = df_merge.loc[df_merge.groupby(['CFM', 'Mesh'])['x-velocity'].idxmax()]
    df_merge = df_merge.drop(['x-coordinate', 'y-coordinate'], axis=1)
    table = pd.pivot_table(df_merge, values='x-velocity', index=['CFM'], columns=['Mesh'])
    table_2 = table
    # table_2 = table.iloc[:,[2,0,1]] # adjust table
    table_2.to_csv(acoil + '_table_for_GCI_ke_M1to4.csv')


def unpack(x):
    if x:
        return x[0]
    return np.nan


def output_full_profile(dg):
    # Getting full profile
    x_half = dg['Normalized Coil Height [-]'].to_numpy().reshape(-1, 1)
    y_half = dg['Air Velocity [m/s]'].to_numpy().reshape(-1, 1)
    y_rev = y_half[::-1]
    x_rev = x_half + x_half[-1]
    y = np.concatenate((y_half, y_rev[1:]), axis=0)
    x = np.concatenate((x_half, x_rev[1:]), axis=0)
    x = x / max(x)
    y[y <= 0] = 0.00001
    df_x = pd.DataFrame(x, columns=['Normalized Coil Height [-]'])
    df_y = pd.DataFrame(y, columns=['Air Velocity [m/s]'])
    df_out = pd.concat([df_x, df_y], axis=1)
    return df_out


def pred_plot(model_input, preds):
    x = model_input['Normalized Coil Height [-]']
    FR = model_input['CFM'][0]
    deg = model_input['angle'][0]
    h = model_input['height'][0]
    fig, ax1 = plt.subplots(figsize = (8,6))
    sns.set(font_scale=2)
    plt.scatter(x, preds)  # predict profile
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax1.set_xlabel('Normalized Coil Height [-]', fontsize = 20)
    ax1.set_ylabel('X-velocity [m/s]', fontsize = 20)
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(axis="y", direction="in")
    plt.title("%0.0fCFM, half angle:$%0.1f^\circ$C, HX height:%0.2f m" % (FR, deg, h), fontsize=20)
    # ax1.set_ylim(-0.25, 2)
    ax1.set_xlim(0, 1)
    plt.legend(['Prediction'], loc='upper right', facecolor="white")
    # plt.savefig('xxx.jpg')
    plt.show()


def pred_plot_compare(model_input, preds, dh):
    x = model_input['Normalized Coil Height [-]']
    FR = model_input['CFM'][0]
    deg = model_input['angle'][0]
    h = model_input['height'][0]
    vals = dh['Air Velocity [m/s]'].to_numpy()
    fig, ax1 = plt.subplots(figsize = (8,6))
    sns.set(font_scale=2)
    plt.scatter(x, preds)  # predict profile
    plt.scatter(dh['Normalized Coil Height [-]'], vals)  # reference profile
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax1.set_xlabel('Normalized Coil Height [-]', fontsize = 20)
    ax1.set_ylabel('X-velocity [m/s]', fontsize = 20)
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(axis="y", direction="in")
    rmse = mean_squared_error(preds, vals, squared=False)
    cvrmse = rmse / np.mean(vals)
    plt.text(0.3, 0.1, '$RMSE$: %.3f \n$CvRMSE$: %.3f' % (rmse, cvrmse), fontsize=20)
    plt.title("%0.0fCFM, half angle:$%0.1f^\circ$C, HX height:%0.2f m" % (FR, deg, h), fontsize=20)
    # ax1.set_ylim(-0.25, 2)
    ax1.set_xlim(0, 1)
    plt.legend(['Prediction', 'Reference ('+str(dh['CFM'][0])+'CFM)'], loc='upper right', facecolor="white")
    plt.legend(['Prediction'], loc='upper right', facecolor="white")
    # plt.savefig('xxx.jpg')
    plt.show()


def ml_cv_plot(index, MAE, result_std):
    print(MAE)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    plt.errorbar(index, MAE, yerr=result_std)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax1.set_xlabel('Methods', fontsize=20)
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=20)
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(axis="y", direction="in")
    plt.title("20 folds cross validation", fontsize=20)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    labels = [item.get_text() for item in ax1.get_xticklabels()]
    # labels[1] = 'LR'
    labels[1] = 'KNN'
    labels[2] = 'DT'
    labels[3] = 'RF'
    labels[4] = 'GP'
    ax1.set_xticklabels(labels)
    # plt.savefig('CV.jpg')
    plt.show()
