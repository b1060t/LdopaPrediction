import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
import os
import os.path
import sys
sys.path.append('..')
from src.utils.data import getDict, writePandas, getPandas, getConfig, writeConfig

def filter_by_threshold(filename, pathlabel, threshold):
    data = getPandas(filename)
    imgs = data[pathlabel].tolist()
    img_list = [nib.load(img).get_fdata() for img in imgs]
    imgs = np.array(img_list)
    mean_img = np.mean(imgs, axis=0)
    mean_img[mean_img < threshold] = 0
    mean_img[mean_img >= threshold] = 1
    img_list = [img * mean_img for img in img_list]
    return mean_img

def genSubcorticalVolume(filename):
    pass