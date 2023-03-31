import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
import os
import os.path
import sys
from nilearn import image
import radiomics
import logging
from radiomics import featureextractor
sys.path.append('..')
from src.utils.data import getDict, writePandas

def genTextureFeature(data, label):
    radiomics.logger.setLevel(logging.ERROR)
    extract = featureextractor.RadiomicsFeatureExtractor()
    extract.loadParams(os.path.join('data', 'config', 'radiomic.yaml'))
    mask_tags = getDict('subcortical_roi')
    def cal_radiomics(path):
        filtered_rst = {}
        for key in mask_tags.keys():
            rst = extract.execute(path, os.path.join('data', 'bin', 'subcortical_roi', key + '.nii'))
            for k, v in rst.items():
                if ('firstorder' in k) or ('glcm' in k) or ('gldm' in k) or ('glrlm' in k) or ('glszm' in k):
                #if ('firstorder' in k):
                    filtered_rst[key + '_' + k] = v
        return filtered_rst
    rsts = list(map(cal_radiomics, data[label]))
    data_radiomic = pd.DataFrame(rsts)
    data_radiomic = data_radiomic.astype(float)
    data_radiomic['KEY'] = data['KEY']
    writePandas(label+'_radiomic', data_radiomic)