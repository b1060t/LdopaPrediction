from process.clinic_data import writeClinic
from process.img_data import imgRedir, preprocFSL
from process.pat_data import combinePatData
import os
os.chdir('..')

preprocFSL()