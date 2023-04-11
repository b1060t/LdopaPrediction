import os
import os.path
import sys
import pandas as pd
sys.path.append('..')
from src.utils.data import getDict, writePandas, getPandas, getConfig

def surfCAT12(filename):
    data = getPandas(filename)
    
    # params
    from nipype.pipeline.engine import Workflow, Node
    from nipype.interfaces.cat12.surface import ExtractAdditionalSurfaceParameters
    from src.interface.cat12 import ExtractROIBasedSurfaceMeasures
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    
    key_list = data['KEY'].tolist()
    
    wf = Workflow(name='cat12surf', base_dir=os.path.abspath('tmp'))
    
    info_src = Node(util.IdentityInterface(fields=['key']), name='info_src')
    info_src.iterables = ('key', key_list)
    
    raw_src = Node(nio.DataGrabber(infields=['key'], outfields=['left_central_file', 'surf_files']), name='raw_src')
    raw_src.inputs.base_directory = os.path.abspath(os.path.join('data', 'subj'))
    raw_src.inputs.sort_filelist = False
    raw_src.inputs.template = '*'
    raw_src.inputs.template_args = {
        'left_central_file': [['key']],
        'surf_files': [['key']],
    }
    raw_src.inputs.field_template = {
        'left_central_file': os.path.join('%s', 'cat12', 'surf', 'lh.central.raw.gii'),
        'surf_files': os.path.join('%s', 'cat12', 'surf', '*.gii'),
    }
    
    surf_extract = Node(ExtractAdditionalSurfaceParameters(), name='surf_extract')
    surf_extract.inputs.depth = True
    surf_extract.inputs.fractal_dimension = True
    
    sink = Node(nio.DataSink(), name='sink')
    sink.inputs.base_directory = os.path.abspath(os.path.join('data', 'subj'))
    sink.inputs.parameterization = False
    
    wf.connect([
        (info_src, raw_src, [('key', 'key')]),
        (raw_src, surf_extract, [('left_central_file', 'left_central_surfaces'),
                                    ('surf_files', 'surface_files')]),
        (info_src, sink, [('key', 'container')]),
        (surf_extract, sink, [('lh_area', 'cat12.surf.@lh_area'),
                              ('lh_depth', 'cat12.surf.@lh_depth'),
                              ('lh_extracted_files', 'cat12.surf.@lh_extracted_files'),
                              ('lh_fractaldimension', 'cat12.surf.@lh_fractaldimension'),
                              ('lh_gmv', 'cat12.surf.@lh_gmv'),
                              ('lh_gyrification', 'cat12.surf.@lh_gyrification'),
                              ('rh_area', 'cat12.surf.@rh_area'),
                              ('rh_depth', 'cat12.surf.@rh_depth'),
                              ('rh_extracted_files', 'cat12.surf.@rh_extracted_files'),
                              ('rh_fractaldimension', 'cat12.surf.@rh_fractaldimension'),
                              ('rh_gmv', 'cat12.surf.@rh_gmv'),
                              ('rh_gyrification', 'cat12.surf.@rh_gyrification')
                              ]),
    ])
    
    wf.run()
    
    surf_path = list(data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'surf')
    for pth in surf_path:
        print('Extracting surface measures from ' + pth)
        extract = ExtractROIBasedSurfaceMeasures()
        extract.inputs.lh_roi_atlas = [os.path.abspath(os.path.join('data', 'bin', 'lh.aparc_a2009s.freesurfer.annot')), os.path.abspath(os.path.join('data', 'bin', 'lh.aparc_HCP_MMP1.freesurfer.annot'))]
        extract.inputs.rh_roi_atlas = [os.path.abspath(os.path.join('data', 'bin', 'rh.aparc_a2009s.freesurfer.annot')), os.path.abspath(os.path.join('data', 'bin', 'rh.aparc_HCP_MMP1.freesurfer.annot'))]
        extract.inputs.lh_surface_measure = [
            os.path.abspath(os.path.join(pth, 'lh.gmv.raw')),
            os.path.abspath(os.path.join(pth, 'lh.area.raw')),
            os.path.abspath(os.path.join(pth, 'lh.depth.raw')),
            os.path.abspath(os.path.join(pth, 'lh.fractaldimension.raw')),
            os.path.abspath(os.path.join(pth, 'lh.gyrification.raw'))
            ]
        extract.run()
        
    from xml.dom import minidom
    label_list = list(data['IMG_ROOT'] + os.sep + 'cat12' + os.sep + 'label' + os.sep + 'catROIs_raw.xml')
    label_info = []
    for path in label_list:
        xmldoc = minidom.parse(path)

        # aparc_a2009s
        aparc_a2009s = xmldoc.getElementsByTagName('aparc_a2009s')[0]
        names = aparc_a2009s.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        names = names[2:]
        thickness = aparc_a2009s.getElementsByTagName('data')[0].getElementsByTagName('thickness')[0]
        thickness = [float(x) for x in thickness.childNodes[0].data[1:-1].split(';')][2:]
        gmv = aparc_a2009s.getElementsByTagName('data')[0].getElementsByTagName('gmv')[0]
        gmv = [float(x) for x in gmv.childNodes[0].data[1:-1].split(';')][2:]
        area = aparc_a2009s.getElementsByTagName('data')[0].getElementsByTagName('area')[0]
        area = [float(x) for x in area.childNodes[0].data[1:-1].split(';')][2:]
        depth = aparc_a2009s.getElementsByTagName('data')[0].getElementsByTagName('depth')[0]
        depth = [float(x) for x in depth.childNodes[0].data[1:-1].split(';')][2:]
        fractaldimension = aparc_a2009s.getElementsByTagName('data')[0].getElementsByTagName('fractaldimension')[0]
        fractaldimension = [float(x) for x in fractaldimension.childNodes[0].data[1:-1].split(';')][2:]
        gyrification = aparc_a2009s.getElementsByTagName('data')[0].getElementsByTagName('gyrification')[0]
        gyrification = [float(x) for x in gyrification.childNodes[0].data[1:-1].split(';')][2:]
        thickness_col = ['aparc_a2009s_' + x + '_thickness' for x in names]
        gmv_col = ['aparc_a2009s_' + x + '_gmv' for x in names]
        area_col = ['aparc_a2009s_' + x + '_area' for x in names]
        depth_col = ['aparc_a2009s_' + x + '_depth' for x in names]
        fractaldimension_col = ['aparc_a2009s_' + x + '_fractaldimension' for x in names]
        gyrification_col = ['aparc_a2009s_' + x + '_gyrification' for x in names]
        rec = pd.Series(thickness+gmv+area+depth+fractaldimension+gyrification,
                        index=thickness_col+gmv_col+area_col+depth_col+fractaldimension_col+gyrification_col)
        
        # aparc_DK40
        aparc_DK40 = xmldoc.getElementsByTagName('aparc_DK40')[0]
        names = aparc_DK40.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        names = names[2:]
        thickness = aparc_DK40.getElementsByTagName('data')[0].getElementsByTagName('thickness')[0]
        thickness = [float(x) for x in thickness.childNodes[0].data[1:-1].split(';')][2:]
        thickness_col = ['aparc_DK40_' + x + '_thickness' for x in names]
        rec = pd.concat([rec, pd.Series(thickness, index=thickness_col)])
        
        # aparc_HCP_MMP1
        aparc_HCP_MMP1 = xmldoc.getElementsByTagName('aparc_HCP_MMP1')[0]
        names = aparc_HCP_MMP1.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        names = names[2:]
        area = aparc_HCP_MMP1.getElementsByTagName('data')[0].getElementsByTagName('area')[0]
        area = [float(x) for x in area.childNodes[0].data[1:-1].split(';')][2:]
        depth = aparc_HCP_MMP1.getElementsByTagName('data')[0].getElementsByTagName('depth')[0]
        depth = [float(x) for x in depth.childNodes[0].data[1:-1].split(';')][2:]
        gmv = aparc_HCP_MMP1.getElementsByTagName('data')[0].getElementsByTagName('gmv')[0]
        gmv = [float(x) for x in gmv.childNodes[0].data[1:-1].split(';')][2:]
        fractaldimension = aparc_HCP_MMP1.getElementsByTagName('data')[0].getElementsByTagName('fractaldimension')[0]
        fractaldimension = [float(x) for x in fractaldimension.childNodes[0].data[1:-1].split(';')][2:]
        gyrification = aparc_HCP_MMP1.getElementsByTagName('data')[0].getElementsByTagName('gyrification')[0]
        gyrification = [float(x) for x in gyrification.childNodes[0].data[1:-1].split(';')][2:]
        area_col = ['aparc_HCP_MMP1_' + x + '_area' for x in names]
        depth_col = ['aparc_HCP_MMP1_' + x + '_depth' for x in names]
        gmv_col = ['aparc_HCP_MMP1_' + x + '_gmv' for x in names]
        fractaldimension_col = ['aparc_HCP_MMP1_' + x + '_fractaldimension' for x in names]
        gyrification_col = ['aparc_HCP_MMP1_' + x + '_gyrification' for x in names]
        rec = pd.concat([rec, pd.Series(area+depth+gmv+fractaldimension+gyrification,
                                        index=area_col+depth_col+gmv_col+fractaldimension_col+gyrification_col)])
        
        label_info.append(rec)
    roi_df = pd.concat([data['KEY'], pd.DataFrame(label_info)], axis=1)
    roi_df = roi_df.drop(['aparc_a2009s_lMedial_wall_thickness',
                          'aparc_a2009s_rMedial_wall_thickness',
                          'aparc_a2009s_lMedial_wall_gmv',
                          'aparc_a2009s_rMedial_wall_gmv',
                          'aparc_a2009s_lMedial_wall_area',
                          'aparc_a2009s_rMedial_wall_area',
                          'aparc_a2009s_lMedial_wall_depth',
                          'aparc_a2009s_rMedial_wall_depth',
                          'aparc_a2009s_lMedial_wall_fractaldimension',
                          'aparc_a2009s_rMedial_wall_fractaldimension',
                          'aparc_a2009s_lMedial_wall_gyrification',
                          'aparc_a2009s_rMedial_wall_gyrification',
                          'aparc_DK40_lcorpuscallosum_thickness',
                          'aparc_DK40_rcorpuscallosum_thickness'], axis=1)
    
    prefix = filename.split('_')[0]
    writePandas(prefix + '_roisurf', roi_df)