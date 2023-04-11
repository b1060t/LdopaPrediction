from process.preproc_img import imgRedir, preprocFSL, preprocCAT12
from src.feature.texture import genTextureFeature, dropByCorrelation
from src.feature.surface import surfCAT12
import os
os.chdir('..')

#preprocCAT12('hc_data')
#preprocFSL('pat_data')
#genTextureFeature('pat_data', 'ANTs_Reg')
#dropByCorrelation('pat_data', 'pat_ANTs_Reg_radiomic', 'CAT_MDS')
surfCAT12('pat_data')