import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
import os
import os.path
import sys
import glob
import networkx as nx
from sklearn.decomposition import FastICA
from scipy.stats import zscore, gaussian_kde
from scipy.special import kl_div
sys.path.append('..')
from src.utils.data import getDict, writePandas, getPandas, getConfig, writeConfig, writeGraph, getGraph, getDataPandas, writeData

def genKLS(filename, roi_tag):
    data = getPandas(filename)
    conf = getConfig('data')
    roi_info = getDict(roi_tag)
    train_idx = conf['indices']['pat']['train']
    test_idx = conf['indices']['pat']['test']
    idxs = train_idx + test_idx
    roi = 'data/bin/aal.nii'
    roi = nib.load(roi).get_fdata()
    kdes = {}
    for idx in idxs:
        print('Processing kde {}'.format(data.iloc[idx]['KEY']))
        raw = data.iloc[idx]['CAT12_GM']
        raw = nib.load(raw).get_fdata()
        if len(raw.shape) == 4:
            raw = raw[:,:,:,0]
        kdes[data.iloc[idx]['KEY']] = {}
        for key in roi_info.keys():
            val = roi_info[key]
            roi_tmp = np.zeros(roi.shape)
            roi_tmp[roi==val] = 1
            rst = np.multiply(roi_tmp, raw)
            rst = rst[rst!=0]
            if len(rst) < 2:
                continue
            kde = gaussian_kde(rst)
            kdes[data.iloc[idx]['KEY']][key] = kde
    x = np.linspace(0.01, 1.01, 100)
    roi_list = roi_info.keys()
    roi_list = list(roi_list)
    for key, val in kdes.items():
        # check file existence
        #if os.path.exists('data/json/graph/{}.json'.format(key)):
            #continue
        print('Processing kls {}'.format(key))
        adj_mat = np.zeros((len(roi_list), len(roi_list)))
        cal_mat = np.zeros((len(roi_list), len(roi_list)))
        for key1 in val.keys():
            kde1 = val[key1]
            for key2 in val.keys():
                if key1 == key2:
                    continue
                if cal_mat[roi_list.index(key1), roi_list.index(key2)] == 1 or cal_mat[roi_list.index(key2), roi_list.index(key1)] == 1:
                    continue
                kde2 = val[key2]
                k1 = [kde1(i) for i in x]
                k2 = [kde2(i) for i in x]
                kl1 = np.sum(kl_div(k1, k2))
                kl2 = np.sum(kl_div(k2, k1))
                js = 0.5 * (kl1 + kl2)
                adj_mat[roi_list.index(key1), roi_list.index(key2)] = js
                adj_mat[roi_list.index(key2), roi_list.index(key1)] = js
                cal_mat[roi_list.index(key1), roi_list.index(key2)] = 1
                cal_mat[roi_list.index(key2), roi_list.index(key1)] = 1
        adj_rst = pd.DataFrame(adj_mat, columns=roi_list, index=roi_list)
        # drop 0?
        #adj_rst = adj_rst[adj_rst.columns[adj_rst.sum(axis=0) != 0]]
        #adj_rst = adj_rst.loc[:, adj_rst.sum(axis=0) != 0]
        #adj_rst = adj_rst.loc[adj_rst.sum(axis=1) != 0, :]
        #adj_rst = adj_rst.fillna(float('inf'))
        adj_rst = np.exp(-adj_rst)
        edge_list = []
        for i in range(adj_rst.shape[0]):
            for j in range(adj_rst.shape[1]):
                if i < j:
                    edge_list.append([adj_rst.index[i], adj_rst.columns[j], adj_rst.iloc[i,j]])
        writeGraph(key, edge_list)

def thresholdGraph(threshold):
    conf = getConfig('data')
    data = getPandas('pat_data')
    train_idx = conf['indices']['pat']['train']
    test_idx = conf['indices']['pat']['test']
    idxs = train_idx + test_idx
    collection = {}
    for idx in idxs:
        key = data.iloc[idx]['KEY']
        edges = getGraph(key)
        edges = [edge for edge in edges if edge[2] > threshold]
        writeGraph(key + '_' + str(threshold), edges)

def genNodalFeature(filename):
    conf = getConfig('data')
    data = getPandas(filename)
    roi_info = getDict('aal')
    roi_list = list(roi_info.keys())
    train_idx = conf['indices']['pat']['train']
    test_idx = conf['indices']['pat']['test']
    idxs = train_idx + test_idx
    degree_criterion = 2 * np.log(len(roi_list))
    thresholds = np.linspace(0.5, 0.01, 50)
    network = {}
    consensus = []
    for idx in idxs:
        print('Processing {}'.format(data.iloc[idx]['KEY']))
        graph = None
        for thres in thresholds:
            key = data.iloc[idx]['KEY']
            full_edges = getGraph(key)
            edges = [edge for edge in full_edges if edge[2] > thres]
            edges = [(e[0], e[1], {'weight': e[2]}) for e in edges]
            g = nx.Graph()
            g.add_nodes_from(roi_list)
            g.add_edges_from(edges)
            avg_degree = np.sum(list(dict(g.degree()).values())) / len(roi_list)
            if avg_degree > degree_criterion:
                graph = g
                break
        if graph is None:
            print('No graph found for {}'.format(key))
            return
        network[key] = {}
        network[key]['KEY'] = key
        connected_nodes = max(nx.connected_components(g), key=len)
        subg = g.subgraph(connected_nodes)
        # Degree
        degrees = dict(graph.degree())
        for roi, degree in degrees.items():
            network[key][roi+'_degree'] = degree
        # Degree Centrality
        dcs = nx.degree_centrality(g)
        dcs = dict(dcs)
        for roi, dc in dcs.items():
            network[key][roi+'_dc'] = dc
        # Betweenness Centrality
        bcs = nx.betweenness_centrality(g, normalized=True)
        bcs = dict(bcs)
        for roi, bc in bcs.items():
            network[key][roi+'_bc'] = bc
        # Nodal Clustering Coefficient
        nccs = nx.clustering(g)
        nccs = dict(nccs)
        for roi, ncc in nccs.items():
            network[key][roi+'_ncc'] = ncc
        # Average Clustering
        cp = nx.average_clustering(g)
        network[key]['cp'] = cp
        # Global Efficiency
        ge = nx.global_efficiency(g)
        network[key]['ge'] = ge
        # Local Efficiency
        le = nx.local_efficiency(g)
        network[key]['le'] = le
        # Characteristic Path Length
        cpl = nx.average_shortest_path_length(subg)
        network[key]['cpl'] = cpl
        # Modularity Score
        comm = nx.community.louvain_communities(g)
        mod = nx.community.modularity(g, comm)
        network[key]['mod'] = mod
        # Sigma
        sigma = nx.sigma(subg, niter=10)
        network[key]['sigma'] = sigma

        # Nodal Efficiency
        nes = {m+'_'+n: 0 if m == n else nx.efficiency(subg, m, n) for m in connected_nodes for n in connected_nodes}
        network[key]['ne'] = np.sum(list(nes.values()))
        #for edge, ne in nes.items():
            #network[key][edge+'_ne'] = ne
        # Shortest Path Length
        spls = {m+'_'+n: 0 if m == n else nx.shortest_path_length(subg, m, n) for m in connected_nodes for n in connected_nodes}
        network[key]['spl'] = np.sum(list(spls.values()))
        #for edge, spl in spls.items():
            #network[key][edge+'_spl'] = spl

        # Consensus Connections
        consensus.append(subg.edges())

    network_df = pd.DataFrame.from_dict(network, orient='index')
    writePandas('pat_nodal', network_df)
 
def genICA(filename):
    data = getPandas(filename)
    conf = getConfig('data')
    train_idx = conf['indices']['pat']['train']
    test_idx = conf['indices']['pat']['test']
    sgm_path = data.iloc[train_idx]['CAT12_T1'].tolist()
    train_sgm_arr = np.array([nib.load(path).get_fdata() for path in sgm_path])
    train_sgm_arr = train_sgm_arr.reshape(train_sgm_arr.shape[0], -1)
    # drop 0, save mask
    mask = np.all(train_sgm_arr==0, axis=0)
    train_sgm_arr = train_sgm_arr[:, ~mask]
    train_sgm_arr = zscore(train_sgm_arr, axis=1)
    ica_transformer = FastICA(n_components=60, random_state=0)
    #ica_transformer.fit_transform(train_sgm_arr)
    ica_transformer.fit(train_sgm_arr)
    # transform both train and test data
    sgm_path = data['CAT12_T1'].tolist()
    sgm_arr = np.array([nib.load(path).get_fdata() for path in sgm_path])
    sgm_arr = sgm_arr.reshape(sgm_arr.shape[0], -1)
    sgm_arr = sgm_arr[:, ~mask]
    sgm_arr = zscore(sgm_arr, axis=1)
    #sgm_ica = np.zeros((sgm_arr.shape[0], 35))
    #for sgm in sgm_arr:
        #print(sgm.shape)
        #ica = ica_transformer.transform(sgm)
        #sgm_ica = np.vstack((sgm_ica, ica))
    sgm_ica = ica_transformer.transform(sgm_arr)
    keys = data['KEY'].tolist()
    ica_df = pd.DataFrame(sgm_ica, columns=['ICA_{}'.format(i+1) for i in range(sgm_ica.shape[1])])
    ica_df['KEY'] = keys
    writePandas('pat_sgm_ica', ica_df)
    return sgm_ica