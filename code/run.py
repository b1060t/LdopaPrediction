from process.preproc_img import imgRedir, preprocFSL, preprocCAT12, preprocANTs, ImageNormalization, ImageMinMaxScale
from src.feature.texture import genTextureFeature, dropByCorrelation, genSubjTextureFeature, genSubjTextureFeatureByROI
from src.feature.surface import surfCAT12
from src.utils.data import getPandas, writePandas, getConfig
from process.preproc_fmriprep import build_pat_bids, build_hc_bids, run_fmriprep, gen_pat_roi
import os
os.chdir('..')
import glob

#gen_pat_roi()
#genSubjTextureFeatureByROI('pat_data', 'fmriprep_native')
#genSubjTextureFeature('pat_data', 'ANTs_Reg_4')
#dropByCorrelation('pat_data', 'pat_ANTs_Reg_4_radiomic', 'CAT')
preprocANTs('pat_data')
#ImageNormalization('pat_data', 'ANTs_Reg', 'ANTs_Reg_Norm', 'PD25/PD25-atlas-mask-1mm.nii.gz')
#ImageNormalization('pat_data', 'ANTs_Reg', 'ANTs_Reg_MinMax', 'PD25/PD25-atlas-mask-1mm.nii.gz')
#dropByCorrelation('pat_data', 'pat_fmriprep_native_radiomic', 'CAT')
#build_pat_bids()
#run_fmriprep()
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
#import nibabel as nib
#data = getPandas('pat_data')
#conf = getConfig('data')
#keys = conf['indices']['pat']['train'] + conf['indices']['pat']['test']
#keys = data.iloc[keys]['KEY'].tolist()
#mismatch_list = []
#for idx, rec in data.iterrows():
    #if rec['KEY'] not in keys:
        #continue
    #img = nib.load(rec['fmriprep_MNI'])
    #if img.shape != (193, 229, 193):
        ##if not os.path.exists(os.path.join('data', 'bids', 'pat_fmriprep', 'sub-{}'.format(rec['KEY']), 'anat', 'sub-' + rec['KEY'] + '_space-MNI152NLin2009cAsym_res-01_desc-preproc_T1w.nii.gz')):
            #mismatch_list.append(rec['KEY'])

#print(len(mismatch_list))
        
#for i, key in enumerate(mismatch_list):
    #print('Processing {}'.format(key))
    #cmd = 'fmriprep-docker data/bids/pat_raw data/bids/pat_fmriprep -i nipreps/fmriprep:latest --mem 8192 --output-space MNI152NLin2009cAsym:res-01 --fs-no-reconall --anat-only --skip_bids_validation'
    #cmd += ' --participant-label {}'.format(key)
    ##if not os.path.exists(os.path.join('data', 'bids', 'pat_fmriprep', 'sub-{}'.format(key))):
    #os.system(cmd)

#3120
#for key in mismatch_list:
    #rec = data[data['KEY'] == key].iloc[0]
    #rec['fmriprep_MNI'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_space-MNI152NLin2009cAsym_res-01_desc-preproc_T1w.nii.gz'
    #rec['fmriprep_MNI_GM'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_space-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz'
    #rec['fmriprep_MNI_WM'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_space-MNI152NLin2009cAsym_res-01_label-WM_probseg.nii.gz'
    #rec['fmriprep_MNI_CSF'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_space-MNI152NLin2009cAsym_res-01_label-CSF_probseg.nii.gz'
    #rec['fmriprep_MNI_brainmask'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_space-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz'
    #rec['fmriprep_native'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_desc-preproc_T1w.nii.gz'
    #rec['fmriprep_native_GM'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_label-GM_probseg.nii.gz'
    #rec['fmriprep_native_WM'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_label-WM_probseg.nii.gz'
    #rec['fmriprep_native_CSF'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_label-CSF_probseg.nii.gz'
    #rec['fmriprep_native_brainmask'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_desc-brain_mask.nii.gz'
    #rec['fmriprep_native2MNI'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    #rec['fmriprep_MNI2native'] = rec['BIDS_ROOT'] + os.sep + 'anat' + os.sep + 'sub-' + rec['KEY'] + '_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'
    #data[data['KEY'] == key] = rec

#writePandas('pat_data', data)