import torch
from torch import nn
from models import resnet
path = '../../../data/bin/resnet_10.pth'

model = resnet.myresnet(
    sample_input_W=193,
    sample_input_H=229,
    sample_input_D=193,
    shortcut_type='B',
    no_cuda=True
)
net_dict = model.state_dict()
pretrain = torch.load(path, map_location='cpu')
# remove 'module.'
pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items() if k[7:] in net_dict.keys()}

net_dict.update(pretrain_dict)
model.load_state_dict(net_dict)

import sys
import os
import nibabel as nib
import numpy as np
import pandas as pd
sys.path.append('../..')
from src.utils.data import writePandas, getPandas, getConfig, getDict
os.chdir('../../..')

data = getPandas('pat_data')
conf = getConfig('data')
train_idx = conf['indices']['pat']['train']
test_idx = conf['indices']['pat']['test']
keys = data['KEY'].values
paths = data['ANTs_Reg'].values

recs = []
for key, path in zip(keys, paths):
    print(key)
    img = nib.load(path).get_fdata()
    i_min = img.min()
    i_std = img.std()
    img = (img - i_min) / i_std
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    img = img.cpu().float()
    output = model(img)
    output = output.cpu().detach().numpy()
    col = ['resnet_'+str(i) for i in range(output.shape[-1])]
    rec = {}
    rec['KEY'] = key
    rec.update(dict(zip(col, output[0])))
    recs.append(rec)
rst = pd.DataFrame(recs)
writePandas('pat_resnet_8_4', rst)

#col = ['resnet_'+str(i) for i in range(512)]
#if os.path.exists('data/json/pat_resnet.json'):
    #rst = getPandas('pat_resnet')
#else:
    #rst = pd.DataFrame(columns=['KEY']+col)
#model.eval()
#for key, path in zip(keys, paths):
    #print(key)
    #if key in rst['KEY'].values:
        #continue
    #img = nib.load(path).get_fdata()
    #i_min = img.min()
    #i_std = img.std()
    #img = (img - i_min) / i_std
    #img = np.expand_dims(img, axis=0)
    #img = np.expand_dims(img, axis=0)
    #img = torch.from_numpy(img)
    #img = img.cpu().float()
    #output = model(img)
    #output = output.cpu().detach().numpy()
    #rec = {}
    #rec['KEY'] = key
    #rec.update(dict(zip(col, output[0])))
    #rst = pd.concat([rst, pd.DataFrame(rec, index=[0])], axis=0)
    #writePandas('pat_resnet', rst)