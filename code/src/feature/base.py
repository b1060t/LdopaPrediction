import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
import os
import os.path
import sys
from nilearn import image
sys.path.append('..')
from src.utils.data import writeData

def genROI():
    mask_path = 'data/bin/PD25-subcortical-1mm.nii'
    os.mkdir(os.path.join('data', 'bin', 'subcortical_roi'))
    mask_template = nib.load(mask_path).get_fdata().astype(int)
    mask_tags = {
        'lRN': [1],
        'lSN': [3],
        'lSTN': [5],
        'lCAU': [7],
        'lPUT': [9],
        'lGPe': [11],
        'lGPi': [13],
        'lTHA': [15],
        'rRN': [2],
        'rSN': [4],
        'rSTN': [6],
        'rCAU': [8],
        'rPUT': [10],
        'rGPe': [12],
        'rGPi': [14],
        'rTHA': [16],
    }
    for tag, labels in mask_tags.items():
        mask = np.full(mask_template.shape, 0)
        for label in labels:
            mask[mask_template == label] = 1
        img = image.new_img_like(mask_path, mask)
        img.to_filename(os.path.join('data', 'bin', 'subcortical_roi', tag + '.nii'))
    writeData('subcortical_roi', mask_tags)