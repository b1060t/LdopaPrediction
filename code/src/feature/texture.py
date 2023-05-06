import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
import os
import os.path
import sys
import radiomics
import logging
from radiomics import featureextractor
sys.path.append('..')
from src.utils.data import getDict, writePandas, getPandas, getConfig, writeConfig

def genTextureFeature(filename, pathlabel):
    data = getPandas(filename)
    radiomics.logger.setLevel(logging.ERROR)
    extract = featureextractor.RadiomicsFeatureExtractor()
    extract.loadParams(os.path.join('config', 'radiomic.yaml'))
    mask_tags = getDict('subcortical_roi')
    def cal_radiomics(path):
        print('Calculating radiomic features for ' + path + '...')
        filtered_rst = {}
        for key in mask_tags.keys():
            rst = extract.execute(path, os.path.join('data', 'bin', 'subcortical_roi', key + '.nii'))
            for k, v in rst.items():
                if ('firstorder' in k) or ('glcm' in k) or ('gldm' in k) or ('glrlm' in k) or ('glszm' in k):
                #if ('firstorder' in k):
                    filtered_rst[key + '_' + k] = v
        return filtered_rst
    rsts = list(map(cal_radiomics, data[pathlabel]))
    data_radiomic = pd.DataFrame(rsts)
    data_radiomic = data_radiomic.astype(float)
    data_radiomic['KEY'] = data['KEY']
    prefix = filename.split('_')[0]
    writePandas(prefix+'_'+pathlabel+'_radiomic', data_radiomic)

def genMalpemTextureFeature(filename, pathlabel):
    pass
    
def dropByCorrelation(data_filename, radiomic_filename, y_label, threshold=0.8):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_selection import r_regression
    from scipy.stats import spearmanr
    mask_tags = getDict('subcortical_roi')
    data_radiomic = getPandas(radiomic_filename)
    data = getPandas(data_filename)
    config = getConfig('data')
    isCont = y_label in config['cont_tags']['y']
    rad = data_radiomic.copy()
    rad = rad.iloc[config['indices']['pat']['train']].drop(['KEY'], axis=1)
    for tag, label in mask_tags.items():
        cor = abs(data_radiomic.filter(like=tag, axis=1).corr())
        print(tag)
        sns.heatmap(cor, cmap=sns.color_palette("coolwarm", as_cmap=True), xticklabels=False, yticklabels=False)
        fig = plt.gcf()
        fig.savefig(os.path.join('data', 'img', 'texture', 'correlation', tag + '.png'))
        fig.clear()
    y = data.iloc[config['indices']['pat']['train']][y_label].to_numpy()
    removal = []
    for tag, labels in mask_tags.items():
        if isCont:
            cor = rad.filter(like=tag, axis=1).corr()
            r = abs(r_regression(rad.filter(like=tag, axis=1), y.ravel()))
            cor = abs(cor)
            size = len(cor)
            for i in range(size):
                col = cor.iloc[:, i]
                for j in range(i):
                    val = col.iloc[j]
                    if val > threshold:
                        if (col.name in removal) or (cor.iloc[:, j].name in removal):
                            continue
                        tmp = col.name if (r[i] < r[j]) else cor.iloc[:, j].name
                        if not (tmp in removal):
                            removal.append(tmp)
        else:
            cor = rad.filter(like=tag, axis=1).corr()
            p = []
            c = []
            for fea_idx in range(len(rad.filter(like=tag, axis=1).columns)):
                rst = spearmanr(rad.filter(like=tag, axis=1).iloc[:,fea_idx], y.ravel())
                c.append(abs(rst.correlation))
                p.append(rst.pvalue)
            cor = abs(cor)
            size = len(cor)
            for i in range(size):
                col = cor.iloc[:, i]
                for j in range(i):
                    val = col.iloc[j]
                    if val > threshold:
                        if (col.name in removal) or (cor.iloc[:, j].name in removal):
                            continue
                        tmp = col.name if (c[i] < c[j]) else cor.iloc[:, j].name
                        if not (tmp in removal):
                            removal.append(tmp)
    data_radiomic = data_radiomic.drop(removal, axis=1)
    name = radiomic_filename + '_' + y_label + '_' + str(threshold)
    writePandas(name, data_radiomic)
    config['features']['texture'].append({
        'image': radiomic_filename,
        'class': y_label,
        'threshold': threshold,
        'path': os.path.join('data', 'json', name + '.json')
    })
    writeConfig('data', config)