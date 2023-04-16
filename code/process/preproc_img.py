import os
import os.path
import pandas as pd
import sys
sys.path.append('..')
from src.utils.data import writePandas, getPandas

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
    smooth.inputs.fwhm = 4
    
    msk = Node(ApplyMask(), name='msk')
    msk.inputs.mask_file = os.path.abspath(os.path.join('data', 'bin', 'brainmask_GMtight.nii'))
    
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