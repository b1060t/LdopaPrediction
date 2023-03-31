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
    
    writePandas('pat_data', data)
    
    return data