import os
import os.path
import pandas as pd
import sys
sys.path.append('..')
from src.utils.data import writePandas

def writeClinic():

    raw_age_at_visit = pd.read_csv(os.path.join('data', 'csv', 'Age_at_visit.csv'))
    raw_u3_score = pd.read_csv(os.path.join('data', 'csv', 'MDS_UPDRS_Part_III_CAL.csv'))
    #Deprecated
    #raw_u3_on_off = pd.read_csv(os.path.join('..', 'data', 'csv', 'MDS-UPDRS_Part_III_ON_OFF_Determination___Dosing.csv'))
    raw_demographic = pd.read_csv(os.path.join('data', 'csv', 'Demographics.csv'))
    raw_img_info = pd.read_csv(os.path.join('data', 'raw', 'img_data.csv'))
    raw_diag = pd.read_csv(os.path.join('data', 'csv', 'PD_Diagnosis_History.csv'))
    raw_ledd = pd.read_csv(os.path.join('data', 'csv', 'LEDD_Concomitant_Medication_Log.csv'))

    def apply_filter(x):
        # MED_ON
        on_score = list(x[x['PDSTATE'] == 'ON']['NP3TOT'])
        if len(on_score) == 0:
            on_score = [None]
        # MED_OFF
        off_score = list(x[x['PDSTATE'] == 'OFF']['NP3TOT'])
        if len(off_score) == 0:
            off_score = [None]
        return pd.Series({'INFODT': list(x['INFODT'])[0], 'NUPDR3OF': off_score[0], 'NUPDR3ON': on_score[0]})

    # Keep duplicate index
    u3_dup_idx = raw_u3_score.duplicated(subset=['PATNO', 'EVENT_ID'], keep=False)
    # Get duplicate records
    u3_rec = raw_u3_score[u3_dup_idx][['PATNO', 'EVENT_ID', 'INFODT', 'PDSTATE', 'NP3TOT']].dropna().reset_index(drop=True)
    # Generate U3 ON/OFF records by PATNO and EVENT_ID
    u3_rec = u3_rec.groupby(['PATNO', 'EVENT_ID']).apply(apply_filter).reset_index().dropna().reset_index(drop=True)
    # Get image id
    image_meta = raw_img_info.rename(columns={'Image Data ID': 'IMG_ID', 'Subject': 'PATNO', 'Visit': 'EVENT_ID'})
    # Merge U3 records and image id
    data = pd.merge(u3_rec, image_meta, on=['PATNO', 'EVENT_ID'])[['PATNO', 'EVENT_ID', 'INFODT', 'NUPDR3OF', 'NUPDR3ON', 'IMG_ID']].reset_index(drop=True)
    # Merge age_at_visit df and main df to extract age by EVENT_ID
    data = pd.merge(data, raw_age_at_visit, on=['PATNO', 'EVENT_ID'])[['PATNO', 'EVENT_ID', 'INFODT', 'NUPDR3OF', 'NUPDR3ON', 'IMG_ID', 'AGE_AT_VISIT']].reset_index(drop=True)
    # Merge demographic df and main df
    data = pd.merge(data, raw_demographic.drop(labels=['EVENT_ID', 'INFODT'], axis=1), on=['PATNO'])[['PATNO', 'EVENT_ID', 'INFODT', 'NUPDR3OF', 'NUPDR3ON', 'IMG_ID', 'AGE_AT_VISIT', 'SEX', 'ORIG_ENTRY']].reset_index(drop=True)

    # Duration calculation function
    def get_duration(rec):
        visit = rec.EVENT_ID
        id = rec.PATNO
        visit = rec.INFODT
        diag = raw_diag[raw_diag['PATNO'] == id]['PDDXDT'].iloc[0]
        visit = visit.split('/')
        diag = diag.split('/')
        return 12 * (int(visit[1]) - int(diag[1])) + int(visit[0]) - int(diag[0])

    # Calculate score
    data['SCORE'] = (data['NUPDR3OF'] - data['NUPDR3ON']) / data['NUPDR3OF']
    # Calculate duration
    data['DURATION'] = data.apply(get_duration, axis=1)
    # Calculate categories
    data['CAT'] = 1 * (data['SCORE'] >= 0.3)
    data['CAT_MDS'] = 1 * (data['SCORE'] >= 0.245)
    # Generate unique key
    data['KEY'] = data['PATNO'].astype(str) + data['EVENT_ID'] + data['IMG_ID']
    # Reformat INFODT
    data['INFODT'] = pd.to_datetime(data['INFODT'])

    # LEDD extraction
    # ???
    ledd_rec = raw_ledd[['PATNO', 'LEDTRT', 'STARTDT', 'STOPDT', 'LEDD']].copy()
    # Convert to date
    ledd_rec['STARTDT'] = pd.to_datetime(ledd_rec['STARTDT'])
    ledd_rec['STOPDT'] = pd.to_datetime(ledd_rec['STOPDT'])
    # Fill blank stop date with current date
    ledd_rec['STOPDT'] = ledd_rec['STOPDT'].fillna(pd.Timestamp.now())
    # Drop duplicate records
    ledd_rec = ledd_rec.dropna().drop_duplicates(subset=['PATNO', 'LEDTRT', 'STARTDT', 'STOPDT', 'LEDD']).reset_index(drop=True)

    from functools import reduce
    def get_ledd(rec):
        date = rec.INFODT
        id = rec.PATNO
        ledd_history = ledd_rec[ledd_rec['PATNO'] == id]
        # Filter by date, records at start date are dropped
        ledd_history = ledd_history[(ledd_history['STARTDT'] < date) & (ledd_history['STOPDT'] >= date)]
        ledd_list = ledd_history['LEDD']
        # Check if value is float
        ledd_isfloat = list(map(lambda x: x.replace('.','',1).isdigit(), ledd_list))
        # Generate string index list
        ledd_notfloat = [not e for e in ledd_isfloat]
        ld = 0
        # Drop records without baseline ld value
        if len(ledd_list[ledd_isfloat]) == 0:
            return None
        # Sum all float values
        ld = float(reduce(lambda x, y: float(x)+float(y), ledd_list[ledd_isfloat]))
        # Return if no inhibitor is used
        if len(ledd_list[ledd_notfloat]) == 0:
            return ld
        # Replace LD in inhibitor string with ld value
        ledd_eval = list(map(lambda s: s.replace('LD', str(ld)), ledd_list[ledd_notfloat]))
        # Calculate inhibitor values
        ledd_eval = list(map(lambda s: s.replace('x', '*'), ledd_eval))
        ledd_eval = list(map(lambda s: float(eval(s)), ledd_eval))
        # Sum all available values
        ld += sum(ledd_eval)
        return ld

    # Get LEDD for all records
    data['LEDD'] = data.apply(get_ledd, axis=1)

    data['INFODT'] = data['INFODT'].astype(str)
    data = data.dropna().reset_index(drop=True)

    writePandas('pat_clinic', data)
    
    return data