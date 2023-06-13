import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
import os
import os.path
import sys
import glob
from sklearn.decomposition import FastICA
from scipy.stats import zscore, gaussian_kde
sys.path.append('..')
from src.utils.data import getDict, writePandas, getPandas, getConfig, writeConfig, writeGraph, getGraph, getDataPandas, writeData

def genKLS(filename, roi_tag):
    data = getPandas(filename)
    conf = getConfig('data')
    roi_info = getDict(roi_tag)
    train_idx = conf['indices']['pat']['train']
    test_idx = conf['indices']['pat']['test']
    idxs = train_idx + test_idx
    roi = 'data/bin/aal3.nii'
    roi = nib.load(roi).get_fdata()
    kdes = {}
    for idx in idxs:
        raw = data.iloc[idx]['CAT12_GM']
        raw = nib.load(raw).get_fdata()
        if len(raw.shape) == 4:
            raw = raw[:,:,:,0]
        kdes[data.iloc[idx]['KEY']] = {}
        for key in roi_info.keys():
            val = roi_info[key]
            roi_tmp = np.zeros(roi.shape)
            roi_tmp[roi==val] = 1
            rst = np.multiply(roi_tmp, raw)
            rst = rst[rst!=0]
            if len(rst) == 0:
                continue
            kde = gaussian_kde(rst)
            kdes[data.iloc[idx]['KEY']][key] = kde
    x = np.linspace(0.01, 1.01, 100)
    roi_list = roi_info.keys()
    roi_list = list(roi_list)
    for key, val in kdes.items():
        adj_mat = np.zeros((len(roi_list), len(roi_list)))
        cal_mat = np.zeros((len(roi_list), len(roi_list)))
        for key1 in val.keys():
            kde1 = val[key1]
            for key2 in val.keys():
                if key1 == key2:
                    continue
                if cal_mat[roi_list.index(key1), roi_list.index(key2)] == 1 or cal_mat[roi_list.index(key2), roi_list.index(key1)] == 1:
                    continue
                kde2 = val[key2]
                kl1 = np.sum(kde1(x) * np.log(kde1(x)/kde2(x)))
                kl2 = np.sum(kde2(x) * np.log(kde2(x)/kde1(x)))
                js = 0.5 * (kl1 + kl2)
                adj_mat[roi_list.index(key1), roi_list.index(key2)] = js
                adj_mat[roi_list.index(key2), roi_list.index(key1)] = js
                cal_mat[roi_list.index(key1), roi_list.index(key2)] = 1
                cal_mat[roi_list.index(key2), roi_list.index(key1)] = 1
        adj_rst = pd.DataFrame(adj_mat, columns=roi_list, index=roi_list)
        adj_rst = adj_rst[adj_rst.columns[adj_rst.sum(axis=0) != 0]]
        adj_rst = adj_rst.loc[:, adj_rst.sum(axis=0) != 0]
        adj_rst = adj_rst[adj_rst.sum(axis=1) != 0, :]
        adj_rst = np.exp(-adj_rst)
        writeGraph(key, adj_rst)
            
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