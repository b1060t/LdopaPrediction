import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
import os
import os.path
import sys
import glob
from sklearn.decomposition import FastICA
from scipy.stats import zscore
sys.path.append('..')
from src.utils.data import getDict, writePandas, getPandas, getConfig, writeConfig

def genICA(filename):
    data = getPandas(filename)
    conf = getConfig('data')
    train_idx = conf['indices']['pat']['train']
    test_idx = conf['indices']['pat']['test']
    sgm_path = data.iloc[train_idx]['CAT12_T1'].tolist()
    train_sgm_arr = np.array([nib.load(path).get_fdata() for path in sgm_path])
    train_sgm_arr = train_sgm_arr.reshape(train_sgm_arr.shape[0], -1)
    # drop 0, save mask
    mask = np.all(train_sgm_arr==0, axis=0)
    train_sgm_arr = train_sgm_arr[:, ~mask]
    train_sgm_arr = zscore(train_sgm_arr, axis=1)
    ica_transformer = FastICA(n_components=60, random_state=0)
    #ica_transformer.fit_transform(train_sgm_arr)
    ica_transformer.fit(train_sgm_arr)
    # transform both train and test data
    sgm_path = data['CAT12_T1'].tolist()
    sgm_arr = np.array([nib.load(path).get_fdata() for path in sgm_path])
    sgm_arr = sgm_arr.reshape(sgm_arr.shape[0], -1)
    sgm_arr = sgm_arr[:, ~mask]
    sgm_arr = zscore(sgm_arr, axis=1)
    #sgm_ica = np.zeros((sgm_arr.shape[0], 35))
    #for sgm in sgm_arr:
        #print(sgm.shape)
        #ica = ica_transformer.transform(sgm)
        #sgm_ica = np.vstack((sgm_ica, ica))
    sgm_ica = ica_transformer.transform(sgm_arr)
    keys = data['KEY'].tolist()
    ica_df = pd.DataFrame(sgm_ica, columns=['ICA_{}'.format(i+1) for i in range(sgm_ica.shape[1])])
    ica_df['KEY'] = keys
    writePandas('pat_sgm_ica', ica_df)
    return sgm_ica