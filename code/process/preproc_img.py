import os
import os.path
import pandas as pd
import sys
sys.path.append('..')
from src.utils.data import writePandas, getPandas, getConfig

def imgRedir():
    
    raw_img_info = pd.read_csv(os.path.join('data', 'raw', 'img_data.csv'))
    image_meta = raw_img_info.rename(columns={'Image Data ID': 'IMG_ID', 'Subject': 'PATNO', 'Visit': 'EVENT_ID'})

    # Get image metadata file path
    xmllist = []
    desclist = []
    for dirpath, dirnames, filenames in os.walk(os.path.join('data', 'raw', 'meta')):
        for filename in filenames:
            if dirpath == os.path.join('data', 'raw', 'meta'):
                desclist.append(os.path.join(dirpath, filename))
            else:
                xmllist.append(os.path.join(dirpath, filename))
            
    # Generate image description dataframe by PATNO, IMG_ID, and SERIES
    descs = []
    from xml.dom import minidom
    for xml in desclist:
        root = minidom.parse(xml).documentElement
        subject = root.getElementsByTagName('subjectIdentifier')[0].childNodes[0].data
        series = root.getElementsByTagName('seriesIdentifier')[0].childNodes[0].data
        image = root.getElementsByTagName('imageUID')[0].childNodes[0].data
        seq = root.getElementsByTagName('description')[0].childNodes[0].data
        protocol = root.getElementsByTagName('protocol')
        rec = {'PATNO': int(subject), 'IMG_ID': 'I'+str(image), 'SERIES': 'S'+str(series)}
        for node in protocol:
            term = node.getAttribute('term')
            value = ''
            if node.childNodes:
                value = node.childNodes[0].data
            rec[term] = value
        descs.append(rec)
    descs = pd.DataFrame(descs)

    # Generate image dataframe by PATNO and IMG_ID
    data = []
    from xml.dom import minidom
    for xml in xmllist:
        root = minidom.parse(xml).documentElement
        subject = root.getElementsByTagName('subject')[0].getAttribute('id')
        study = root.getElementsByTagName('study')[0].getAttribute('uid')
        series = root.getElementsByTagName('series')[0].getAttribute('uid')
        image = root.getElementsByTagName('image')[0].getAttribute('uid')
        relative_path = os.path.join('data', 'raw', 'img', xml.split(os.sep)[3], xml.split(os.sep)[4], xml.split(os.sep)[5])
        nii_dir = os.listdir(relative_path)[0]
        nii_name = os.listdir(os.path.join(relative_path, nii_dir))[0]
        relative_path = os.path.join(relative_path, nii_dir, nii_name)
        data.append({'PATNO': int(subject), 'IMG_ID': str(image), 'SERIES': str(series), 'IMG_PATH': str(relative_path)})
    data = pd.DataFrame(data)

    data = pd.merge(data, descs, on=['PATNO', 'IMG_ID', 'SERIES'], how='left')
    data = pd.merge(data, image_meta, on=['PATNO', 'IMG_ID'], how='left')
    #data = data[['PATNO', 'SERIES', 'IMG_ID', 'EVENT_ID', 'Group', 'Sex', 'Age', 'Acq Date', 'IMG_PATH']]
    data['KEY'] = data['PATNO'].astype(str) + data['EVENT_ID'].astype(str) + data['IMG_ID'].astype(str)
    
    writePandas('img_raw', data)
    
    return data


def mvRaw(meta):
    import shutil
    for index, row in meta.iterrows():
        print(row['KEY'])
        # if directory exists, skip it
        if os.path.exists(os.path.join('data', 'subj', row['KEY'], 'raw')):
            continue
        os.makedirs(os.path.join('data', 'subj', row['KEY'], 'raw'))
        shutil.copy(row['IMG_PATH'], os.path.join('data', 'subj', row['KEY'], 'raw', 'raw.nii'))
        

def preprocANTs(file_name):
    from nipype.pipeline.engine import Workflow, Node
    from nipype import Function
    import nipype.interfaces.fsl as fsl
    fsl.FSLCommand.set_default_output_type('NIFTI')
    import nipype.interfaces.utility as util
    from nipype.interfaces.fsl import BET, FAST, ApplyMask
    from nipype.interfaces.ants import RegistrationSynQuick, ApplyTransforms, Atropos
    from nipype.interfaces.spm import Smooth
    import nipype.interfaces.io as nio
    
    data = getPandas(file_name)
    data = data.iloc[:1]

    key_list = data['KEY'].tolist()

    wf = Workflow(name='ants5', base_dir=os.path.abspath('tmp'))

    info_src = Node(util.IdentityInterface(fields=['key']), name='info_src')
    info_src.iterables = ('key', key_list)

    raw_src = Node(nio.DataGrabber(infields=['key'], outfields=['raw']), name='raw_src')
    raw_src.inputs.base_directory = os.path.abspath(os.path.join('data', 'subj'))
    raw_src.inputs.sort_filelist = False
    raw_src.inputs.template = '*'
    raw_src.inputs.template_args = {
        'raw': [['key']]
    }
    raw_src.inputs.field_template = {
        'raw': os.path.join('%s', 'raw', 'raw.nii')
    }

    reg = Node(RegistrationSynQuick(), name='reg')
    reg.inputs.fixed_image = os.path.abspath(os.path.join('PD25', 'PD25-T1MPRAGE-template-1mm.nii.gz'))
    reg.inputs.num_threads = 1

    mat = Node(util.Merge(2), name='mat')

    transform = Node(ApplyTransforms(), name='transform')
    transform.inputs.input_image = os.path.abspath(os.path.join('PD25', 'PD25-subcortical-1mm.nii.gz'))
    transform.inputs.interpolation = 'NearestNeighbor'
    transform.inputs.invert_transform_flags = [True, False]

    atropos = Node(Atropos(likelihood_model='Gaussian', save_posteriors=True, mrf_smoothing_factor=0.2, mrf_radius=[1, 1, 1], n_iterations=5, convergence_threshold=0.000001, posterior_formulation='Socrates', use_mixture_model_proportions=True), name='atropos')
    #atropos = Node(Atropos(save_posteriors=True), name='atropos')
    atropos.inputs.dimension = 3
    atropos.inputs.mask_image = os.path.abspath(os.path.join('PD25', 'PD25-atlas-mask-1mm.nii.gz'))
    atropos.inputs.number_of_tissue_classes = 3
    atropos.inputs.initialization = 'PriorProbabilityImages'
    atropos.inputs.prior_image = os.path.abspath(os.path.join('data', 'bin', 'tpm%02d.nii.gz'))
    atropos.inputs.prior_weighting = 0.55

    gm_selector = Node(util.Select(index=[0]), name='gm_selector')

    smth = Node(fsl.Smooth(), name='smth')
    smth.inputs.fwhm = 4
    
    transform_pos = Node(ApplyTransforms(), name='transform_pos')
    transform_pos.inputs.invert_transform_flags = [True, False]

    sinker = Node(nio.DataSink(), name='sinker')
    sinker.inputs.base_directory = os.path.abspath(os.path.join('data', 'subj'))
    sinker.inputs.parameterization = False
    sinker.inputs.regexp_substitutions = [
        ('raw_brain', 'brain'),
        ('transformWarped', 'reg'),
        ('pve_0', 'csf'),
        ('pve_1', 'gm'),
        ('pve_2', 'wm'),
    ]

    wf.connect([
        (info_src, raw_src, [('key', 'key')]),
        (raw_src, reg, [('raw', 'moving_image')]),
        (info_src, sinker, [('key', 'container')]),
        (reg, mat, [('inverse_warp_field', 'in2'),
                     ('out_matrix', 'in1')]),
        (reg, atropos, [('warped_image', 'intensity_images')]),
        (raw_src, transform, [('raw', 'reference_image')]),
        (mat, transform, [('out', 'transforms')]),
        (atropos, gm_selector, [('posteriors', 'inlist')]),
        (gm_selector, smth, [('out', 'in_file')]),
        (reg, sinker, [('warped_image', 'ants5.@warped_image'),
                       ('inverse_warp_field', 'ants5.@inverse_warp_field'),
                       ('forward_warp_field', 'ants5.@forward_warp_field'),
                       ('out_matrix', 'ants5.@out_matrix'),]),
        (transform, sinker, [('output_image', 'ants5.@transformed_image')]),
        (atropos, sinker, [('classified_image', 'ants5.@classified_image'),
                           ('posteriors', 'ants5.@posteriors')]),
        (smth, sinker, [('smoothed_file', 'ants5.@smoothed_file')]),
    ])

    wf.run()
    
    data['ANTs_Reg_5'] = data['IMG_ROOT'] + os.sep + 'ants5' + os.sep + 'reg.nii.gz'
    data['ANTs_5_inverse'] = data['IMG_ROOT'] + os.sep + 'ants5' + os.sep + 'transform1InverseWarp.nii.gz'
    data['ANTs_5_forward'] = data['IMG_ROOT'] + os.sep + 'ants5' + os.sep + 'transform1Warp.nii.gz'
    data['ANTs_5_affine'] = data['IMG_ROOT'] + os.sep + 'ants5' + os.sep + 'transform0GenericAffine.mat'
    data['ANTs_5_native_subcortical_ROI'] = data['IMG_ROOT'] + os.sep + 'ants5' + os.sep + 'PD25-subcortical-1mm_trans.nii.gz'

    #if seg:
        #mask = 'PD25/PD25-atlas-mask-1mm.nii.gz'
        #tpms = 'data/bin/tpm%d.nii.gz'
        #mask = os.path.abspath(mask)
        #tpms = os.path.abspath(tpms)
        #regs = data['ANTs_Reg_5'].tolist()
        #roots = data['IMG_ROOT'].tolist()
        #regs = [os.path.abspath(reg) for reg in regs]
        #roots = [os.path.abspath(root) for root in roots]
        #roots = [os.path.join(root, 'ants5') for root in roots]
        #for i in range(len(regs)):
            #os.chdir(roots[i])
            #print('Working on %s' % regs[i])
            #cmd = 'antsAtroposN4.sh -d 3 -a %s -x %s -o seg -c 3 -s nii -p %s -y 2 -y 3 -w 0.25' % (regs[i], mask, tpms)
            #os.system(cmd)
    #antsAtroposN4.sh -d 3 -a testWarped.nii.gz -x ../PD25/PD25-atlas-mask-1mm.nii.gz -o seg -c 3 -s nii -p tpm%d.nii.gz -y 2 -y 3 -w 0.25 
    
    #writePandas(file_name, data)

def preprocAtropos(file_name):
    data = getPandas(file_name)

    def atropos(rec):
        reg_file = os.path.abspath(rec['ANTs_Reg'])
        img_root = os.path.abspath(rec['IMG_ROOT'])

def preprocFSL(file_name):
    from nipype.pipeline.engine import Workflow, Node
    from nipype import Function
    import nipype.interfaces.fsl as fsl
    fsl.FSLCommand.set_default_output_type('NIFTI')
    import nipype.interfaces.utility as util
    from nipype.interfaces.fsl import BET, FAST, ApplyMask
    from nipype.interfaces.ants import RegistrationSynQuick
    from nipype.interfaces.spm import Smooth
    import nipype.interfaces.io as nio
    
    data = getPandas(file_name)

    def gm_extract(pve_files):
        return pve_files[1]

    key_list = data['KEY'].tolist()

    wf = Workflow(name='preproc', base_dir=os.path.abspath('tmp'))

    info_src = Node(util.IdentityInterface(fields=['key']), name='info_src')
    info_src.iterables = ('key', key_list)

    raw_src = Node(nio.DataGrabber(infields=['key'], outfields=['raw']), name='raw_src')
    raw_src.inputs.base_directory = os.path.abspath(os.path.join('data', 'subj'))
    raw_src.inputs.sort_filelist = False
    raw_src.inputs.template = '*'
    raw_src.inputs.template_args = {
        'raw': [['key']]
    }
    raw_src.inputs.field_template = {
        'raw': os.path.join('%s', 'raw', 'raw.nii')
    }

    bet = Node(BET(), name='bet')
    bet.inputs.robust = True

    reg = Node(RegistrationSynQuick(), name='reg')
    reg.inputs.fixed_image = os.path.abspath(os.path.join('data', 'bin', 'template.nii.gz'))
    reg.inputs.num_threads = 16

    fslseg = Node(FAST(), name='fslseg')
    fslseg.inputs.output_type = 'NIFTI'
    fslseg.inputs.segments = True
    fslseg.inputs.probability_maps = True
    fslseg.inputs.number_classes = 3

    gmextract = Node(Function(input_names=['pve_files'], output_names=['gm_file'], function=gm_extract), name='gmextract')

    smooth = Node(Smooth(), name='smooth')
    smooth.inputs.fwhm = 4

    msk = Node(ApplyMask(), name='msk')
    msk.inputs.mask_file = os.path.abspath(os.path.join('data', 'bin', 'mask.nii'))

    sinker = Node(nio.DataSink(), name='sinker')
    sinker.inputs.base_directory = os.path.abspath(os.path.join('data', 'subj'))
    sinker.inputs.parameterization = False
    sinker.inputs.regexp_substitutions = [
        ('raw_brain', 'brain'),
        ('transformWarped', 'reg'),
        ('pve_0', 'csf'),
        ('pve_1', 'gm'),
        ('pve_2', 'wm'),
    ]

    wf.connect([
        (info_src, raw_src, [('key', 'key')]),
        (raw_src, bet, [('raw', 'in_file')]),
        (bet, reg, [('out_file', 'moving_image')]),
        (reg, fslseg, [('warped_image', 'in_files')]),
        (fslseg, gmextract, [('partial_volume_files', 'pve_files')]),
        (gmextract, smooth, [('gm_file', 'in_files')]),
        (smooth, msk, [('smoothed_files', 'in_file')]),
        (info_src, sinker, [('key', 'container')]),
        (bet, sinker, [('out_file', 'fsl.@out_file')]),
        (reg, sinker, [('warped_image', 'fsl.@warped_image')]),
        (fslseg, sinker, [('partial_volume_files', 'fsl.@partial_volume_files')]),
        (msk, sinker, [('out_file', 'fsl.@masked_smoothed_file')]),
    ])

    wf.run()
    
    data['ANTs_Reg'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'reg.nii.gz'
    data['FSL_GM'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'reg_gm.nii'
    data['FSL_WM'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'reg_wm.nii'
    data['FSL_CSF'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'reg_csf.nii'
    data['FSL_SGM'] = data['IMG_ROOT'] + os.sep + 'fsl' + os.sep + 'sreg_gm_masked.nii'
    
    writePandas(file_name, data)

def preprocCAT12(filename):
    from nipype.pipeline.engine import Workflow, Node
    from nipype.interfaces.cat12.preprocess import CAT12Segment
    from nipype.interfaces.spm.preprocess import Smooth
    import nipype.interfaces.fsl as fsl
    fsl.FSLCommand.set_default_output_type('NIFTI')
    from nipype.interfaces.fsl import ApplyMask
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    
    data = getPandas(filename)

    key_list = data['KEY'].tolist()
    
    wf = Workflow(name='cat12segment', base_dir=os.path.abspath('tmp'))
    
    info_src = Node(util.IdentityInterface(fields=['key']), name='info_src')
    info_src.iterables = ('key', key_list)
    
    raw_src = Node(nio.DataGrabber(infields=['key'], outfields=['raw']), name='raw_src')
    raw_src.inputs.base_directory = os.path.abspath(os.path.join('data', 'subj'))
    raw_src.inputs.sort_filelist = False
    raw_src.inputs.template = '*'
    raw_src.inputs.template_args = {
        'raw': [['key']]
    }
    raw_src.inputs.field_template = {
        'raw': os.path.join('%s', 'raw', 'raw.nii')
    }
    
    seg = Node(CAT12Segment(), name='seg')
    #seg.inputs.own_atlas = [os.path.abspath(os.path.join('data', 'bin', 'aal3.nii'))]
    
    smooth = Node(Smooth(), name='smooth')
    smooth.inputs.fwhm = 8
    
    msk = Node(ApplyMask(), name='msk')
    msk.inputs.mask_file = os.path.abspath(os.path.join('data', 'bin', 'brainmask_GM.nii'))
    
    sink = Node(nio.DataSink(), name='sink')
    sink.inputs.base_directory = os.path.abspath(os.path.join('data', 'subj'))
    sink.inputs.parameterization = False
    
    wf.connect([
        (info_src, raw_src, [('key', 'key')]),
        (raw_src, seg, [('raw', 'in_files')]),
        (info_src, sink, [('key', 'container')]),
        (seg, smooth, [('gm_modulated_image', 'in_files')]),
        (smooth, msk, [('smoothed_files', 'in_file')]),
        (msk, sink, [('out_file', 'cat12.@masked_smoothed_file')]),
        (seg, sink, [
            ('gm_modulated_image', 'cat12.mri.@gm'),
            ('wm_modulated_image', 'cat12.mri.@wm'),
            ('csf_modulated_image', 'cat12.mri.@csf'),
            ('mri_images', 'cat12.mri.@mri_images'),
            ('label_files', 'cat12.label.@label_files'),
            ('label_roi', 'cat12.label.@label_roi'),
            ('label_rois', 'cat12.label.@label_rois'),
            ('lh_central_surface', 'cat12.surf.@lh_central_surface'),
            ('lh_sphere_surface', 'cat12.surf.@lh_sphere_surface'),
            ('rh_central_surface', 'cat12.surf.@rh_central_surface'),
            ('rh_sphere_surface', 'cat12.surf.@rh_sphere_surface'),
            ('report_files', 'cat12.report.@report_files'),
            ('report', 'cat12.report.@report'),
            ('surface_files', 'cat12.surf.@surface_files'),
            ]),
    ])
    
    wf.run()
    
    # IQR check
    iqr_list = []
    from xml.dom import minidom
    report_list = list(data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'report' + os.sep + 'cat_raw.xml')
    for report in report_list:
        root = minidom.parse(report).documentElement
        iqr_str = root.getElementsByTagName('catlog')[0].getElementsByTagName('item')[-5].childNodes[0].data
        iqr_str = iqr_str.split(' ')[4][:-1]
        iqr_list.append({'IQR': float(iqr_str)})
    iqr_list = pd.DataFrame(iqr_list)
    data = pd.concat([data, iqr_list], axis=1)
    #data = data[data['IQR'] >= 70].reset_index(drop=True)
    
    data['CAT12_GM'] = data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'mri' + os.sep + 'mwp1raw.nii'
    data['CAT12_SGM'] = data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'smwp1raw_masked.nii'
    data['CAT12_WM'] = data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'mri' + os.sep + 'mwp2raw.nii'
    data['CAT12_CSF'] = data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'mri' + os.sep + 'mwp3raw.nii'
    
    # Whole brain volume: saved in xxx_data.json
    from xml.dom import minidom
    report_list = list(data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'report' + os.sep + 'cat_raw.xml')
    vol_list = []
    for report in report_list:
        root = minidom.parse(report).documentElement
        tiv_str = root.getElementsByTagName('subjectmeasures')[1].getElementsByTagName('vol_TIV')[0].childNodes[0].data
        vol_str = root.getElementsByTagName('subjectmeasures')[1].getElementsByTagName('vol_abs_CGW')[0].childNodes[0].data
        tiv = float(tiv_str)
        gm = float(vol_str.split(' ')[1])
        wm = float(vol_str.split(' ')[2])
        vol_list.append({'TIV': tiv, 'GM_VOL': gm, 'WM_VOL': wm})
    vol_list = pd.DataFrame(vol_list)
    data = pd.concat([data, vol_list], axis=1)
    
    # ROI volume: saved in xxx_roivol.json
    label_list = list(data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'label' + os.sep + 'catROI_raw.xml')
    label_info = []
    for path in label_list:
        xmldoc = minidom.parse(path)
        
        # Cobra
        cobra = xmldoc.getElementsByTagName('cobra')[0]
        names = cobra.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        vgm = cobra.getElementsByTagName('data')[0].getElementsByTagName('Vgm')[0]
        vgm = [float(x) for x in vgm.childNodes[0].data[1:-1].split(';')]
        vwm = cobra.getElementsByTagName('data')[0].getElementsByTagName('Vwm')[0]
        vwm = [float(x) for x in vwm.childNodes[0].data[1:-1].split(';')]
        gm_col = ['cobra_gm_' + x for x in names]
        wm_col = ['cobra_wm_' + x for x in names]
        rec = pd.Series(vgm + vwm, index=gm_col+wm_col)
        
        # Hammers
        hammers = xmldoc.getElementsByTagName('hammers')[0]
        names = hammers.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        vcsf = hammers.getElementsByTagName('data')[0].getElementsByTagName('Vcsf')[0]
        vcsf = [float(x) for x in vcsf.childNodes[0].data[1:-1].split(';')]
        vgm = hammers.getElementsByTagName('data')[0].getElementsByTagName('Vgm')[0]
        vgm = [float(x) for x in vgm.childNodes[0].data[1:-1].split(';')]
        vwm = hammers.getElementsByTagName('data')[0].getElementsByTagName('Vwm')[0]
        vwm = [float(x) for x in vwm.childNodes[0].data[1:-1].split(';')]
        csf_col = ['hammers_csf_' + x for x in names]
        gm_col = ['hammers_gm_' + x for x in names]
        wm_col = ['hammers_wm_' + x for x in names]
        rec = pd.concat([rec, pd.Series(vcsf + vgm + vwm, index=csf_col+gm_col+wm_col)])
        
        # Lpba40
        lpba40 = xmldoc.getElementsByTagName('lpba40')[0]
        names = lpba40.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        vgm = lpba40.getElementsByTagName('data')[0].getElementsByTagName('Vgm')[0]
        vgm = [float(x) for x in vgm.childNodes[0].data[1:-1].split(';')]
        vwm = lpba40.getElementsByTagName('data')[0].getElementsByTagName('Vwm')[0]
        vwm = [float(x) for x in vwm.childNodes[0].data[1:-1].split(';')]
        gm_col = ['lpba40_gm_' + x for x in names]
        wm_col = ['lpba40_wm_' + x for x in names]
        rec = pd.concat([rec, pd.Series(vgm + vwm, index=gm_col+wm_col)])
        
        # Neuromorphometrics
        neuromorphometrics = xmldoc.getElementsByTagName('neuromorphometrics')[0]
        names = neuromorphometrics.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        vcsf = neuromorphometrics.getElementsByTagName('data')[0].getElementsByTagName('Vcsf')[0]
        vcsf = [float(x) for x in vcsf.childNodes[0].data[1:-1].split(';')]
        vgm = neuromorphometrics.getElementsByTagName('data')[0].getElementsByTagName('Vgm')[0]
        vgm = [float(x) for x in vgm.childNodes[0].data[1:-1].split(';')]
        vwm = neuromorphometrics.getElementsByTagName('data')[0].getElementsByTagName('Vwm')[0]
        vwm = [float(x) for x in vwm.childNodes[0].data[1:-1].split(';')]
        csf_col = ['neuromorphometrics_csf_' + x for x in names]
        gm_col = ['neuromorphometrics_gm_' + x for x in names]
        wm_col = ['neuromorphometrics_wm_' + x for x in names]
        rec = pd.concat([rec, pd.Series(vcsf + vgm + vwm, index=csf_col+gm_col+wm_col)])
        
        # Suit
        suit = xmldoc.getElementsByTagName('suit')[0]
        names = suit.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        vgm = suit.getElementsByTagName('data')[0].getElementsByTagName('Vgm')[0]
        vgm = [float(x) for x in vgm.childNodes[0].data[1:-1].split(';')]
        vwm = suit.getElementsByTagName('data')[0].getElementsByTagName('Vwm')[0]
        vwm = [float(x) for x in vwm.childNodes[0].data[1:-1].split(';')]
        gm_col = ['suit_gm_' + x for x in names]
        wm_col = ['suit_wm_' + x for x in names]
        rec = pd.concat([rec, pd.Series(vgm + vwm, index=gm_col+wm_col)])
        
        # Thalamic nuclei
        thalamic_nuclei = xmldoc.getElementsByTagName('thalamic_nuclei')[0]
        names = thalamic_nuclei.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        vgm = thalamic_nuclei.getElementsByTagName('data')[0].getElementsByTagName('Vgm')[0]
        vgm = [float(x) for x in vgm.childNodes[0].data[1:-1].split(';')]
        gm_col = ['thalamic_nuclei_gm_' + x for x in names]
        rec = pd.concat([rec, pd.Series(vgm, index=gm_col)])
        
        # Thalamus
        thalamus = xmldoc.getElementsByTagName('thalamus')[0]
        #ids = [int(x) for x in thalamus.getElementsByTagName('ids')[0].childNodes[0].data[1:-1].split(';')]
        names = thalamus.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        vgm = thalamus.getElementsByTagName('data')[0].getElementsByTagName('Vgm')[0]
        vgm = [float(x) for x in vgm.childNodes[0].data[1:-1].split(';')]
        gm_col = ['thalamus_gm_' + x for x in names]
        rec = pd.concat([rec, pd.Series(vgm, index=gm_col)])

        label_info.append(rec)
    roi_df = pd.concat([data['KEY'], pd.DataFrame(label_info)], axis=1)
    prefix = filename.split('_')[0]
    writePandas(prefix + '_roivol', roi_df)
    
    writePandas(filename, data)

def ImageNormalization(filename, pathlabel, newlabel, mask):
    data = getPandas(filename)
    conf = getConfig('data')
    used_inds = conf['indices']['pat']['train'] + conf['indices']['pat']['test']
    proc_data = data.loc[used_inds].reset_index(drop=True)
    img_roots = proc_data['IMG_ROOT'].tolist()
    img_list = proc_data[pathlabel].tolist()
    import nibabel as nib
    import numpy as np
    from numpy import ma
    mask = nib.load(mask).get_data()
    mask = mask > 0
    for i in range(len(img_list)):
        print('Normalizing image ' + str(i) + ' of ' + str(len(img_list)))
        img = nib.load(img_list[i])
        affine = img.affine
        img = img.get_fdata()
        masked_img = ma.masked_array(img, mask=~mask)
        mean = np.mean(masked_img)
        std = np.std(masked_img)
        img = (img - mean) / std
        img = nib.Nifti1Image(img, affine)
        path = os.path.join(img_roots[i], newlabel)
        nib.save(img, path)
    img_roots = data['IMG_ROOT'].tolist()
    img_path = [os.path.join(x, newlabel + '.nii') for x in img_roots]
    data[newlabel] = img_path
    writePandas(filename, data)

def ImageMinMaxScale(filename, pathlabel, newlabel, mask):
    data = getPandas(filename)
    conf = getConfig('data')
    used_inds = conf['indices']['pat']['train'] + conf['indices']['pat']['test']
    proc_data = data.loc[used_inds].reset_index(drop=True)
    img_roots = proc_data['IMG_ROOT'].tolist()
    img_list = proc_data[pathlabel].tolist()
    import nibabel as nib
    import numpy as np
    from numpy import ma
    mask = nib.load(mask).get_data()
    mask = mask > 0
    for i in range(len(img_list)):
        print('Normalizing image ' + str(i) + ' of ' + str(len(img_list)))
        img = nib.load(img_list[i])
        affine = img.affine
        img = img.get_fdata()
        masked_img = ma.masked_array(img, mask=~mask)
        min_val = np.min(masked_img)
        max_val = np.max(masked_img)
        img = (img - min_val) / (max_val - min_val)
        img = nib.Nifti1Image(img, affine)
        path = os.path.join(img_roots[i], newlabel)
        nib.save(img, path)
    img_roots = data['IMG_ROOT'].tolist()
    img_path = [os.path.join(x, newlabel + '.nii') for x in img_roots]
    data[newlabel] = img_path
    writePandas(filename, data)
    

def ImageStatistics(filename, pathlabel, mask):
    data = getPandas(filename)
    conf = getConfig('data')
    used_inds = conf['indices']['pat']['train'] + conf['indices']['pat']['test']
    proc_data = data.loc[used_inds].reset_index(drop=True)
    img_roots = proc_data['IMG_ROOT'].tolist()
    img_list = proc_data[pathlabel].tolist()
    import nibabel as nib
    import numpy as np
    from numpy import ma
    mask = nib.load(mask).get_data()
    mask = mask > 0
    stat = []
    for i in range(len(img_list)):
        img = nib.load(img_list[i])
        img = img.get_fdata()
        masked_img = ma.masked_array(img, mask=~mask)
        mean_val = np.mean(masked_img)
        std_val = np.std(masked_img)
        min_val = np.min(masked_img)
        max_val = np.max(masked_img)
        stat.append({
            'KEY': proc_data['KEY'][i],
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        })
    stat_df = pd.DataFrame(stat)
    return stat_df