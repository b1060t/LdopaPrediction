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
    #ANTs
    data['AGE_CORRECTED_SGM'] = data['IMG_ROOT'] + os.sep + 'agecorrected.nii'
    data['ANTs_Reg'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'reg.nii.gz'
    data['FSL_GM'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'reg_gm.nii'
    data['FSL_WM'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'reg_wm.nii'
    data['FSL_CSF'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'reg_csf.nii'
    data['FSL_SGM'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'sreg_gm_masked.nii'
    #CAT12
    data['CAT12_GM'] = data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'mri' + os.sep + 'mwp1raw.nii'
    data['CAT12_SGM'] = data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'mri' + os.sep + 'smwp1raw_masked.nii'
    data['CAT12_WM'] = data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'mri' + os.sep + 'mwp2raw.nii'
    data['CAT12_CSF'] = data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'mri' + os.sep + 'mwp3raw.nii'
    #fmriprep (training/test set only to save space)
    data['BIDS_ROOT'] = data['KEY'].apply(lambda s: os.path.join('data', 'bids', 'pat_fmriprep', 'sub-{}'.format(s)))
    data['fmriprep_MNI'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'.format(data['KEY'])
    data['fmriprep_MNI_GM'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz'.format(data['KEY'])
    data['fmriprep_MNI_WM'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_space-MNI152NLin2009cAsym_label-WM_probseg.nii.gz'.format(data['KEY'])
    data['fmriprep_MNI_CSF'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz'.format(data['KEY'])
    data['fmriprep_MNI_brainmask'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(data['KEY'])
    data['fmriprep_native'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_desc-preproc_T1w.nii.gz'.format(data['KEY'])
    data['fmriprep_native_GM'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_label-GM_probseg.nii.gz'.format(data['KEY'])
    data['fmriprep_native_WM'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_label-WM_probseg.nii.gz'.format(data['KEY'])
    data['fmriprep_native_CSF'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_label-CSF_probseg.nii.gz'.format(data['KEY'])
    data['fmriprep_native_brainmask'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_desc-brain_mask.nii.gz'.format(data['KEY'])
    data['fmriprep_native2MNI'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'.format(data['KEY'])
    data['fmriprep_MNI2native'] = data['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-{}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'.format(data['KEY'])
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