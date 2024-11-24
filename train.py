import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from hyperparameters import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.metrics import Metric

sig_df = pd.read_csv('data/Jewel.csv')
bg_df = pd.read_csv('data/Pythia.csv')
herwig_df = pd.read_csv('data/Herwig.csv')
hydrocolbt_df = pd.read_csv('data/CoLBTHydro.csv')

sig_X = sig_df.dropna().iloc[:,0:-1].values
sig_Y = sig_df.dropna().iloc[:,-1].values

bg_X = bg_df.dropna().iloc[:,0:-1].values
bg_Y = bg_df.dropna().iloc[:,-1].values

herwig_X = herwig_df.dropna().iloc[:,0:-1].values
herwig_Y = herwig_df.dropna().iloc[:,-1].values

hydrocolbt_X = hydrocolbt_df.dropna().iloc[:,0:-1].values
hydrocolbt_Y = hydrocolbt_df.dropna().iloc[:,-1].values

sig_X_train, sig_X_test, sig_Y_train, sig_Y_test = train_test_split(sig_X, sig_Y, train_size=0.9, shuffle=False)
bg_X_train, bg_X_test, bg_Y_train, bg_Y_test = train_test_split(bg_X, bg_Y, train_size=0.9, shuffle=False)
herwig_X_train, herwig_X_test, herwig_Y_train, herwig_Y_test = train_test_split(herwig_X, herwig_Y, train_size=0.9, shuffle=False)
hydrocolbt_X_train, hydrocolbt_X_test, hydrocolbt_Y_train, hydrocolbt_Y_test = train_test_split(hydrocolbt_X, hydrocolbt_Y, train_size=0.9, shuffle=False)

X_train = np.concatenate((sig_X_train,  bg_X_train, herwig_X_train, hydrocolbt_X_train), axis=0)
Y_train = np.concatenate((sig_Y_train,  bg_Y_train, herwig_Y_train, hydrocolbt_Y_train), axis=0)
X_test = np.concatenate((sig_X_test, bg_X_test, herwig_X_test, hydrocolbt_X_test), axis=0)
Y_test = np.concatenate((sig_Y_test, bg_Y_test, herwig_Y_test, hydrocolbt_Y_test), axis=0)

# Fix random seed to ensure reproducibility 
seed = 6

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = TabNetRegressor(device_name=device, n_a=n_a, n_d=n_d, n_steps=n_steps, gamma=gamma, seed=seed)
# model.load_model('model/model.zip')

model.fit(X_train, Y_train.reshape(-1, 1), eval_set=[(X_test, Y_test.reshape(-1, 1))], batch_size=16384, max_epochs=150)
model.save_model('model/model')