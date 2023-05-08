import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
import os
import os.path
import sys
import glob
import radiomics
import logging
from radiomics import featureextractor
sys.path.append('..')
from src.utils.data import getDict, writePandas, getPandas, getConfig, writeConfig

def genMalpemFeature(filename):
    data = getPandas(filename)
    def extract_malpem(rec):
        print('Extracting malpem features for ' + rec['KEY'] + '...')
        root = rec['IMG_ROOT']
        key = rec['KEY']
        csv_path_list = glob.glob('{}/malpem/*/report/raw_Report_MALPEM_Report.csv'.format(root))
        if len(csv_path_list) > 0:
            csv_path = csv_path_list[0]
            vol_data = pd.read_csv(csv_path, index_col=None, skiprows=7, header=None)
            vol_data = pd.Series(vol_data[2].values, index=vol_data[1]).to_frame().T
            vol_data['KEY'] = key
            return vol_data
        else:
            return pd.DataFrame()
    # combine series to dataframe
    rsts = data.apply(extract_malpem, axis=1)
    malpem_data = pd.concat(rsts.tolist(), ignore_index=True)
    print(malpem_data.shape)
    writePandas('pat_malpem', malpem_data)

def genMalpemTexture(filename):
    data = getPandas(filename)
    radiomics.logger.setLevel(logging.ERROR)
    extract = featureextractor.RadiomicsFeatureExtractor()
    extract.loadParams(os.path.join('config', 'radiomic.yaml'))
