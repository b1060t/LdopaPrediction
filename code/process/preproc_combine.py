import os
import os.path
import pandas as pd
import sys
sys.path.append('..')
from src.utils.data import writePandas, getPandas

def combinePatData():
    img_meta = getPandas('img_raw')
    pat_clinic = getPandas('pat_clinic')
    
    data = pd.merge(pat_clinic, img_meta, on=['KEY', 'PATNO', 'EVENT_ID', 'IMG_ID'], how='left').dropna().drop(columns=['Age', 'Sex', 'IMG_PATH'])
    data['IMG_ROOT'] = data['KEY'].apply(lambda s: os.path.join('data', 'subj', s))
    data.rename(columns={'AGE_AT_VISIT': 'AGE'}, inplace=True)
    writePandas('pat_data', data)
    return data

def genHCData():
    meta = getPandas('img_raw')
    hc_meta = meta[meta['Group'] == 'Control']
    pd.options.mode.chained_assignment = None
    hc_meta['IMG_ROOT'] = hc_meta['KEY'].apply(lambda s: os.path.join('data', 'subj', s))
    hc_meta['Sex'] = 1 * (hc_meta['Sex'] == 'M')
    hc_meta.rename(columns={'Age': 'AGE', 'Sex': 'SEX'}, inplace=True)
    writePandas('hc_data', hc_meta)
    return hc_meta