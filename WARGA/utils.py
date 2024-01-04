# -*- coding: utf-8 -*-
"""
Created on Nov 1 2023

@author: Haiyun Wang
"""


import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import scanpy as sc
import anndata
from scipy.spatial import distance_matrix
import os
import matplotlib.pyplot as plt
import scanpy as sc
import gudhi
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import matplotlib
matplotlib.use('TkAgg')




def preprocess_graph(adj):
    # This code is based on https://github.com/zfjsail/gae-pytorch
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def PAGA_trajectory_inference_and_umap_visualization(adata, args):
    umap_pearlst = anndata.AnnData(adata.obsm['PearlST_embed'])
    umap_pearlst.obs['pred_label'] = list(adata.obs['pred_label'])
    sc.pp.neighbors(umap_pearlst, n_neighbors=15)
    sc.tl.umap(umap_pearlst)
    sc.tl.paga(umap_pearlst, groups='pred_label')

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sc.pl.umap(umap_pearlst, color='pred_label', ax=axs[0], show=False, size=30, legend_fontsize=12)
    axs[0].spines['right'].set_visible(False)  # 去掉边框
    axs[0].spines['top'].set_visible(False)  # 去掉边框
    axs[0].spines['left'].set_visible(False)  # 去掉边框
    axs[0].spines['bottom'].set_visible(False)  # 去掉边框
    axs[0].get_yaxis().set_visible(False)
    axs[0].get_xaxis().set_visible(False)
    axs[0].set_title('UMAP visualization', fontsize=14)

    sc.pl.paga(umap_pearlst, color='pred_label', ax=axs[1], show=False)
    axs[1].spines['right'].set_visible(False)  # 去掉边框
    axs[1].spines['top'].set_visible(False)  # 去掉边框
    axs[1].spines['left'].set_visible(False)  # 去掉边框
    axs[1].spines['bottom'].set_visible(False)  # 去掉边框
    axs[1].get_yaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].set_title('PAGA inference', fontsize=14)
    # save plot
    plt.savefig(str(args.outPath) + '/UMAP and trajectory inference visualization.pdf', dpi=300)




def graph_alpha(spatial_locs, n_neighbors):
    """
    Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
    :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
    :type adata: class:`anndata.annData`
    :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph based on Alpha Complex
    :type n_neighbors: int, optional, default: 10
    :return: a spatial neighbor graph
    :rtype: class:`scipy.sparse.csr_matrix`
    """
    A_knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
    estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
    spatial_locs_list = spatial_locs.tolist()
    n_node = len(spatial_locs_list)
    alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from([i for i in range(n_node)])
    for s in skeleton:
        if len(s[0]) == 2:
            initial_graph.add_edge(s[0][0], s[0][1])

    extended_graph = nx.Graph()
    extended_graph.add_nodes_from(initial_graph)
    extended_graph.add_edges_from(initial_graph.edges)

    # Remove self edges
    for i in range(n_node):
        try:
            extended_graph.remove_edge(i, i)
        except:
            pass

    return nx.to_scipy_sparse_array(extended_graph, format='csr')



def prepare_figure(rsz=4., csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Prepare the figure and axes given the configuration
    :param rsz: row size of the figure in inches, default: 4.0
    :type rsz: float, optional
    :param csz: column size of the figure in inches, default: 4.0
    :type csz: float, optional
    :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type wspace: float, optional
    :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type hspace: float, optional
    :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
    :type left: float, optional
    :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
    :type right: float, optional
    :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
    :type bottom: float, optional
    :param top: the top position of the subplots of the figure in fraction, default: 0.9
    :type top: float, optional
    """
    fig, axs = plt.subplots(1, 1, figsize=(csz, rsz))
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)
    return fig, axs



def pseudo_Spatiotemporal_Map(adata_all, pSM_values_save_filepath="./pSM_values.tsv", n_neighbors=20, resolution=1.0):
    """
    Perform pseudo-Spatiotemporal Map for ST data
    :param pSM_values_save_filepath: the default save path for the pSM values
    :type pSM_values_save_filepath: class:`str`, optional, default: "./pSM_values.tsv"
    :param n_neighbors: The size of local neighborhood (in terms of number of neighboring data
    points) used for manifold approximation. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html` for detail
    :type n_neighbors: int, optional, default: 20
    :param resolution: A parameter value controlling the coarseness of the clustering.
    Higher values lead to more clusters. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html` for detail
    :type resolution: float, optional, default: 1.0
    """
    error_message = "No embedding found, please ensure you have run train() method before calculating pseudo-Spatiotemporal Map!"
    max_cell_for_subsampling = 5000
    try:
        print("Performing pseudo-Spatiotemporal Map")
        adata = anndata.AnnData(adata_all.obsm['spaceflow_emb'])
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
        sc.tl.paga(adata)
        if adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = adata.X
        else:
            indices = np.arange(adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = adata.X[selected_ind, :]
        sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
        adata.uns['iroot'] = np.argmax(sum_dists)
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        pSM_values = adata.obs['dpt_pseudotime'].to_numpy()
        save_dir = os.path.dirname(pSM_values_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt(pSM_values_save_filepath, pSM_values, fmt='%.5f', header='', footer='', comments='')
        print(
            f"pseudo-Spatiotemporal Map(pSM) calculation complete, pSM values of cells or spots saved at {pSM_values_save_filepath}!")
        adata_all.obsm['pSM_values'] = pSM_values
    except NameError:
        print(error_message)
    except AttributeError:
        print(error_message)



def plot_pSM(adata, pSM_figure_save_filepath="./pseudo-Spatiotemporal-Map.pdf", colormap='roma', scatter_sz=1., rsz=4.,
             csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Plot the domain segmentation for ST data in spatial
    :param pSM_figure_save_filepath: the default save path for the figure
    :type pSM_figure_save_filepath: class:`str`, optional, default: "./Spatiotemporal-Map.pdf"
    :param colormap: The colormap to use. See `https://www.fabiocrameri.ch/colourmaps-userguide/` for name list of colormaps
    :type colormap: str, optional, default: roma
    :param scatter_sz: The marker size in points**2
    :type scatter_sz: float, optional, default: 1.0
    :param rsz: row size of the figure in inches, default: 4.0
    :type rsz: float, optional
    :param csz: column size of the figure in inches, default: 4.0
    :type csz: float, optional
    :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type wspace: float, optional
    :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type hspace: float, optional
    :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
    :type left: float, optional
    :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
    :type right: float, optional
    :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
    :type bottom: float, optional
    :param top: the top position of the subplots of the figure in fraction, default: 0.9
    :type top: float, optional
    """
    error_message = "No pseudo Spatiotemporal Map data found, please ensure you have run the pseudo_Spatiotemporal_Map() method."
    try:
        fig, ax = prepare_figure(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right,
                                      bottom=bottom, top=top)
        x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
        st = ax.scatter(x, y, s=scatter_sz, c=adata.obsm['pSM_values'], cmap='summer', marker=".")
        ax.invert_yaxis()
        clb = fig.colorbar(st)
        clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title("SpaceFlow PSM", fontsize=14)
        ax.set_facecolor("none")
        save_dir = os.path.dirname(pSM_figure_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(pSM_figure_save_filepath, dpi=300)
        print(f"Plotting complete, pseudo-Spatiotemporal Map figure saved at {pSM_figure_save_filepath} !")
        plt.close('all')
    except NameError:
        print(error_message)
    except AttributeError:
        print(error_message)

