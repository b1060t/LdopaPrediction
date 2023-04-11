import os
import os.path
import sys
import pandas as pd
sys.path.append('..')
from src.utils.data import getDict, writePandas, getPandas, getConfig

def surfCAT12(filename):
    data = getPandas(filename).iloc[:1]
    
    # thickness
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
        thickness_col = ['aparc_a2009s_' + x for x in names]
        rec = pd.Series(thickness, index=thickness_col)
        
        # aparc_DK40
        aparc_DK40 = xmldoc.getElementsByTagName('aparc_DK40')[0]
        names = aparc_DK40.getElementsByTagName('names')[0].getElementsByTagName('item')
        names = [x.childNodes[0].data for x in names]
        names = names[2:]
        thickness = aparc_DK40.getElementsByTagName('data')[0].getElementsByTagName('thickness')[0]
        thickness = [float(x) for x in thickness.childNodes[0].data[1:-1].split(';')][2:]
        thickness_col = ['aparc_DK40_' + x for x in names]
        rec = pd.concat([rec, pd.Series(thickness, index=thickness_col)])
        
        label_info.append(rec)
    roi_df = pd.concat([data['KEY'], pd.DataFrame(label_info)], axis=1)
    roi_df = roi_df.drop(['aparc_a2009s_lMedial_wall', 'aparc_a2009s_rMedial_wall', 'aparc_DK40_lcorpuscallosum', 'aparc_DK40_rcorpuscallosum'], axis=1)
    
    # params
    from nipype.pipeline.engine import Workflow, Node
    from nipype.interfaces.cat12.surface import ExtractAdditionalSurfaceParameters, ExtractROIBasedSurfaceMeasures
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
    
    lh_merge = Node(util.Merge(5), name='lh_merge')
    rh_merge = Node(util.Merge(5), name='rh_merge')
    lrh_merge = Node(util.Merge(2), name='lrh_merge')
    
    roi_extract = Node(ExtractROIBasedSurfaceMeasures(), name='roi_extract')
    roi_extract.inputs.lh_roi_atlas = os.path.abspath(os.path.join('data', 'bin', 'lh.aparc_a2009s.freesurfer.annot'))
    roi_extract.inputs.rh_roi_atlas = os.path.abspath(os.path.join('data', 'bin', 'rh.aparc_a2009s.freesurfer.annot'))
    
    sink = Node(nio.DataSink(), name='sink')
    sink.inputs.base_directory = os.path.abspath(os.path.join('data', 'subj'))
    sink.inputs.parameterization = False
    
    wf.connect([
        (info_src, raw_src, [('key', 'key')]),
        (raw_src, surf_extract, [('left_central_file', 'left_central_surfaces'),
                                    ('surf_files', 'surface_files')]),
        (info_src, sink, [('key', 'container')]),
        (surf_extract, lh_merge, [('lh_area', 'in1'),
                                  ('lh_depth', 'in2'),
                                  ('lh_fractaldimension', 'in3'),
                                  ('lh_gyrification', 'in4'),
                                  ('lh_gmv', 'in5')]),
        (surf_extract, rh_merge, [('rh_area', 'in1'),
                                  ('rh_depth', 'in2'),
                                  ('rh_fractaldimension', 'in3'),
                                  ('rh_gyrification', 'in4'),
                                  ('rh_gmv', 'in5')]),
        (lh_merge, lrh_merge, [('out', 'in1')]),
        (rh_merge, lrh_merge, [('out', 'in2')]),
        (lh_merge, roi_extract, [('out', 'lh_surface_measure')]),
        #(rh_merge, roi_extract, [('out', 'rh_surface_measure')]),
        #(lrh_merge, roi_extract, [('out', 'surface_files')]),
        (roi_extract, sink, [('label_files', 'cat12.surf.@label_files')]),
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
    
    prefix = filename.split('_')[0]
    writePandas(prefix + '_roisurf', roi_df)