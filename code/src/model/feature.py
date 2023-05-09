import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA, FastICA
import sys
sys.path.append('..')
from src.utils.data import getPandas

def load_img(rec, img_path_tag):
    img_data = np.array(nib.load(rec[img_path_tag]).get_fdata())
    return img_data

def load_imgs(df, img_path_tag):
    return df.apply(lambda d: load_img(d, img_path_tag), axis=1)

def PCA_fit_transform(vox, params):
    pca = PCA(n_components=params['n_components'], random_state=params['pca_random_state'])
    features = pca.fit_transform(vox)
    features = pd.DataFrame(features, columns=['PCA_{}'.format(i+1) for i in range(features.shape[1])])
    return features, pca

def PCA_transform(vox, pca):
    features = pca.transform(vox)
    features = pd.DataFrame(features, columns=['PCA_{}'.format(i+1) for i in range(features.shape[1])])
    return features

def load_radiomics(data, train_idx, test_idx, params):
    df_radiomic = getPandas(params['json_tag'])
    df_radiomic = df_radiomic.drop(['KEY'], axis=1)
    radiomic_train = df_radiomic.iloc[train_idx]
    radiomic_test = df_radiomic.iloc[test_idx]
    return radiomic_train, radiomic_test

def load_volume(data, train_idx, test_idx, params):
    df_volume = data[params['volume_tag']]
    volume_train = df_volume.iloc[train_idx]
    volume_test = df_volume.iloc[test_idx]
    return volume_train, volume_test

def load_roivol(data, train_idx, test_idx, params):
    df_roivol = getPandas(params['json_tag'])
    # only use columns contains hammer
    #df_roivol = df_roivol[df_roivol.columns[df_roivol.columns.str.contains('thalamus_gm')]]
    df_roivol = df_roivol.drop(['KEY'], axis=1)
    roivol_train = df_roivol.iloc[train_idx]
    roivol_test = df_roivol.iloc[test_idx]
    return roivol_train, roivol_test

def load_surface(data, train_idx, test_idx, params):
    df_surface = getPandas(params['json_tag'])
    #df_surface = df_surface[df_surface.columns[df_surface.columns.str.contains('HCP_MMP1')]]
    df_surface = df_surface.drop(['KEY'], axis=1)
    surface_train = df_surface.iloc[train_idx]
    surface_test = df_surface.iloc[test_idx]
    return surface_train, surface_test

def load_malpemvol(data, train_idx, test_idx, params):
    df_malpemvol = getPandas(params['json_tag'])
    df_malpemvol = df_malpemvol.drop(['Background'], axis=1)
    train_keys = data.iloc[train_idx]['KEY'].tolist()
    test_keys = data.iloc[test_idx]['KEY'].tolist()
    malpemvol_train = pd.DataFrame()
    for key in train_keys:
        malpemvol_train = malpemvol_train.append(df_malpemvol[df_malpemvol['KEY'] == key])
    malpemvol_test = pd.DataFrame()
    for key in test_keys:
        malpemvol_test = malpemvol_test.append(df_malpemvol[df_malpemvol['KEY'] == key]) 
    malpemvol_train = malpemvol_train.drop(['KEY'], axis=1)
    malpemvol_test = malpemvol_test.drop(['KEY'], axis=1)
    return malpemvol_train, malpemvol_test

def load_vox_ica(data, train_idx, test_idx, params):
    df_vox_ica = getPandas(params['json_tag'])
    df_vox_ica = df_vox_ica.drop(['KEY'], axis=1)
    vox_ica_train = df_vox_ica.iloc[train_idx]
    vox_ica_test = df_vox_ica.iloc[test_idx]
    return vox_ica_train, vox_ica_test

def gen_voxel_ica_online(data, train_idx, test_idx, params):
    vox_path = data.iloc[train_idx][params['tag']].tolist()
    train_vox_arr = np.array([nib.load(path).get_fdata() for path in vox_path])
    train_vox_arr = train_vox_arr.reshape(train_vox_arr.shape[0], -1)
    # drop 0, save mask
    mask = np.all(train_vox_arr==0, axis=0)
    train_vox_arr = train_vox_arr[:, ~mask]
    train_vox_arr = zscore(train_vox_arr, axis=1)
    ica_transformer = FastICA(n_components=params['n_components'], random_state=0)
    ica_transformer.fit_transform(train_vox_arr)
    ica_transformer.fit(train_vox_arr)
    # transform both train and test data
    vox_path = data[params['tag']].tolist()
    vox_arr = np.array([nib.load(path).get_fdata() for path in vox_path])
    vox_arr = vox_arr.reshape(vox_arr.shape[0], -1)
    vox_arr = vox_arr[:, ~mask]
    vox_arr = zscore(vox_arr, axis=1)
    vox_ica = ica_transformer.transform(vox_arr)
    vox_ica_train = pd.DataFrame(vox_ica[train_idx], columns=['ICA_{}'.format(i+1) for i in range(vox_ica.shape[1])])
    vox_ica_test = pd.DataFrame(vox_ica[test_idx], columns=['ICA_{}'.format(i+1) for i in range(vox_ica.shape[1])])
    return vox_ica_train, vox_ica_test

def gen_feature_ica_online(data, train_idx, test_idx, params):
    features = getPandas(params['json_tag'])
    if params['use_key']:
        train_keys = data.iloc[train_idx]['KEY'].tolist()
        test_keys = data.iloc[test_idx]['KEY'].tolist()
        train_feature = pd.DataFrame()
        for key in train_keys:
            train_feature = train_feature.append(features[features['KEY'] == key])
        test_feature = pd.DataFrame()
        for key in test_keys:
            test_feature = test_feature.append(features[features['KEY'] == key])
    else:
        train_feature = features.iloc[train_idx]
        test_feature = features.iloc[test_idx]
    train_feature = train_feature.drop(['KEY'], axis=1)
    test_feature = test_feature.drop(['KEY'], axis=1)
    train_feature_arr = np.array(train_feature.values.tolist())
    train_feature_arr = zscore(train_feature_arr, axis=1)
    ica_transformer = FastICA(n_components=params['n_components'], random_state=0)
    ica_transformer.fit(train_feature_arr)
    if params['use_key']:
        test_feature_arr = np.array(test_feature.values.tolist())
        test_feature_arr = zscore(test_feature_arr, axis=1)
        feature_ica_train = ica_transformer.transform(train_feature_arr)
        feature_ica_test = ica_transformer.transform(test_feature_arr)
        feature_ica_train = pd.DataFrame(feature_ica_train, columns=['ICA_{}'.format(i+1) for i in range(feature_ica_test.shape[1])])
        feature_ica_test = pd.DataFrame(feature_ica_test, columns=['ICA_{}'.format(i+1) for i in range(feature_ica_test.shape[1])])
        return feature_ica_train, feature_ica_test
    else:
        feature_arr = np.array(features.drop(['KEY'], axis=1).values.tolist())
        feature_arr = zscore(feature_arr, axis=1)
        feature_ica = ica_transformer.transform(feature_arr)
        feature_ica_train = pd.DataFrame(feature_ica[train_idx], columns=['ICA_{}'.format(i+1) for i in range(feature_ica.shape[1])])
        feature_ica_test = pd.DataFrame(feature_ica[test_idx], columns=['ICA_{}'.format(i+1) for i in range(feature_ica.shape[1])])
        return feature_ica_train, feature_ica_test

def gen_voxel_pca_online(data, train_idx, test_idx, params):
    vox_path = data.iloc[train_idx][params['tag']].tolist()
    train_vox_arr = np.array([nib.load(path).get_fdata() for path in vox_path])
    train_vox_arr = train_vox_arr.reshape(train_vox_arr.shape[0], -1)
    # drop 0, save mask
    mask = np.all(train_vox_arr==0, axis=0)
    train_vox_arr = train_vox_arr[:, ~mask]
    train_vox_arr = zscore(train_vox_arr, axis=1)
    pca_transformer = PCA(n_components=params['n_components'], random_state=0)
    pca_transformer.fit_transform(train_vox_arr)
    pca_transformer.fit(train_vox_arr)
    # transform both train and test data
    vox_path = data[params['tag']].tolist()
    vox_arr = np.array([nib.load(path).get_fdata() for path in vox_path])
    vox_arr = vox_arr.reshape(vox_arr.shape[0], -1)
    vox_arr = vox_arr[:, ~mask]
    vox_arr = zscore(vox_arr, axis=1)
    vox_pca = pca_transformer.transform(vox_arr)
    vox_pca_train = pd.DataFrame(vox_pca[train_idx], columns=['PCA_{}'.format(i+1) for i in range(vox_pca.shape[1])])
    vox_pca_test = pd.DataFrame(vox_pca[test_idx], columns=['PCA_{}'.format(i+1) for i in range(vox_pca.shape[1])])
    return vox_pca_train, vox_pca_test

def gen_feature_pca_online(data, train_idx, test_idx, params):
    features = getPandas(params['json_tag'])
    if params['use_key']:
        train_keys = data.iloc[train_idx]['KEY'].tolist()
        test_keys = data.iloc[test_idx]['KEY'].tolist()
        train_feature = pd.DataFrame()
        for key in train_keys:
            train_feature = train_feature.append(features[features['KEY'] == key])
        test_feature = pd.DataFrame()
        for key in test_keys:
            test_feature = test_feature.append(features[features['KEY'] == key])
    else:
        train_feature = features.iloc[train_idx]
        test_feature = features.iloc[test_idx]
    train_feature = train_feature.drop(['KEY'], axis=1)
    test_feature = test_feature.drop(['KEY'], axis=1)
    train_feature_arr = np.array(train_feature.values.tolist())
    train_feature_arr = zscore(train_feature_arr, axis=1)
    pca_transformer = PCA(n_components=params['n_components'], random_state=0)
    pca_transformer.fit(train_feature_arr)
    if params['use_key']:
        test_feature_arr = np.array(test_feature.values.tolist())
        test_feature_arr = zscore(test_feature_arr, axis=1)
        feature_pca_train = pca_transformer.transform(train_feature_arr)
        feature_pca_test = pca_transformer.transform(test_feature_arr)
        feature_pca_train = pd.DataFrame(feature_pca_train, columns=['PCA_{}'.format(i+1) for i in range(feature_pca_test.shape[1])])
        feature_pca_test = pd.DataFrame(feature_pca_test, columns=['PCA_{}'.format(i+1) for i in range(feature_pca_test.shape[1])])
        return feature_pca_train, feature_pca_test
    else:
        feature_arr = np.array(features.drop(['KEY'], axis=1).values.tolist())
        feature_arr = zscore(feature_arr, axis=1)
        feature_pca = pca_transformer.transform(feature_arr)
        feature_pca_train = pd.DataFrame(feature_pca[train_idx], columns=['PCA_{}'.format(i+1) for i in range(feature_pca.shape[1])])
        feature_pca_test = pd.DataFrame(feature_pca[test_idx], columns=['PCA_{}'.format(i+1) for i in range(feature_pca.shape[1])])
        return feature_pca_train, feature_pca_test

Feature_LUT = {
    'ica_voxel_online': gen_voxel_ica_online,
    'ica_feature_online': gen_feature_ica_online,
    'pca_voxel_online': gen_voxel_pca_online,
    'pca_feature_online': gen_feature_pca_online,
    't1_radiomic': load_radiomics,
    't1_radiomic_full': load_radiomics,
    'gm_radiomic': load_radiomics,
    'tiv_gmv': load_volume,
    'roi_volume': load_roivol,
    'surf_info': load_surface,
    'malpem_vol': load_malpemvol
}