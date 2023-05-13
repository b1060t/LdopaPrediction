import os
import os.path
import pandas as pd
import sys
import nibabel as nib
sys.path.append('..')
from src.utils.data import writePandas, getPandas, getConfig

def run_fmriprep():
    # !!! have to process subjects one by one to avoid segmentation fault
    data = getPandas('pat_data')
    conf = getConfig('data')
    train_inds = conf['indices']['pat']['train']
    test_inds = conf['indices']['pat']['test']
    data = data.loc[train_inds + test_inds].reset_index(drop=True)
    keys = data['KEY'].values
    for i, key in enumerate(keys):
        print('Processing {}'.format(key))
        cmd = 'fmriprep-docker data/bids/pat_raw data/bids/pat_fmriprep -i nipreps/fmriprep:latest --mem 8192 --output-space MNI152NLin2009cAsym --fs-no-reconall --anat-only --skip_bids_validation'
        cmd += ' --participant-label {}'.format(key)
        if not os.path.exists(os.path.join('data', 'bids', 'pat_fmriprep', 'sub-{}'.format(key))):
            os.system(cmd)

def build_pat_bids():
    data = getPandas('pat_data')
    conf = getConfig('data')
    train_inds = conf['indices']['pat']['train']
    test_inds = conf['indices']['pat']['test']
    data = data.loc[train_inds + test_inds].reset_index(drop=True)
    keys = data['KEY'].values
    raws = data['IMG_ROOT'].values
    raws = [os.path.join(raw, 'raw', 'raw.nii') for raw in raws]
    for i, key in enumerate(keys):
        #key = ''.join([c for c in key if c.isdigit()])
        print('Processing {}'.format(key))
        raw = raws[i]
        os.makedirs(os.path.join('data', 'bids', 'pat_raw', 'sub-{}'.format(key), 'anat'), exist_ok=True)
        nii_file = nib.load(raw)
        data = nii_file.get_fdata()
        header = nii_file.header
        if len(data.shape) == 4:
            new_data = data[:, :, :, 0]
        new_header = header.copy()
        new_header.set_data_shape(new_data.shape)
        new_nii_file = nib.Nifti1Image(new_data, nii_file.affine, new_header)
        nib.save(new_nii_file, os.path.join('data', 'bids', 'pat_raw', 'sub-{}'.format(key), 'anat', 'sub-{}_T1w.nii'.format(key)))

def build_hc_bids():
    data = getPandas('hc_data')
    keys = data['KEY'].values
    raws = data['IMG_ROOT'].values
    raws = [os.path.join(raw, 'raw', 'raw.nii') for raw in raws]
    for i, key in enumerate(keys):
        #key = ''.join([c for c in key if c.isdigit()])
        print('Processing {}'.format(key))
        raw = raws[i]
        os.makedirs(os.path.join('data', 'bids', 'hc_raw', 'sub-{}'.format(key), 'anat'), exist_ok=True)
        nii_file = nib.load(raw)
        data = nii_file.get_fdata()
        header = nii_file.header
        if len(data.shape) == 4:
            new_data = data[:, :, :, 0]
        new_header = header.copy()
        new_header.set_data_shape(new_data.shape)
        new_nii_file = nib.Nifti1Image(new_data, nii_file.affine, new_header)
        nib.save(new_nii_file, os.path.join('data', 'bids', 'hc_raw', 'sub-{}'.format(key), 'anat', 'sub-{}_T1w.nii'.format(key)))