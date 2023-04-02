from process.clinic_data import writeClinic
from process.img_data import imgRedir, preprocFSL, preprocCAT12
from process.pat_data import combinePatData
import os
os.chdir('..')

#preprocCAT12('hc_data')
preprocFSL('pat_data')