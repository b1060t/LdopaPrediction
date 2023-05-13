from process.preproc_img import imgRedir, preprocFSL, preprocCAT12
from src.feature.texture import genTextureFeature, dropByCorrelation
from src.feature.surface import surfCAT12
from src.utils.data import getPandas, writePandas, getConfig
from process.preproc_fmriprep import build_pat_bids, build_hc_bids, run_fmriprep
import os
os.chdir('..')
import glob

#build_pat_bids()
run_fmriprep()
#preprocCAT12('pat_data')
#preprocFSL('pat_data')
#genTextureFeature('pat_data', 'ANTs_Reg')
#dropByCorrelation('pat_data', 'pat_ANTs_Reg_radiomic', 'CAT_MDS')
#surfCAT12('pat_data')
#genTextureFeature('pat_data', 'ANTs_Reg')

#data = getPandas('pat_data')
#conf = getConfig('data')
#
#keylist = data.iloc[conf['indices']['pat']['train'] + conf['indices']['pat']['test']]['KEY'].tolist()
#
#def malpem(rec):
#    print('Processing {}'.format(rec['KEY']))
#    root = rec['IMG_ROOT']
#    key = rec['KEY']
#    cmd = 'malpem-proot -i {}/raw/raw.nii -o {}/malpem -t 8'.format(root, root)
#    if key in keylist:
#        if not glob.glob('{}/malpem/*/raw_Report.pdf'.format(root)):
#            print('not exist')
#            os.system(cmd)
#        
#data.apply(malpem, axis=1)