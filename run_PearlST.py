# -*- coding: utf-8 -*-
"""
Created on Nov 1 2023

@author: Haiyun Wang
"""
import matplotlib.pyplot as plt
import stlearn as st
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from utilities import parameter_setting
from histology_image_feature_extraction.image_processing import tiling
from histology_image_feature_extraction.image_SSL import train_simCLR_sImage, extract_representation_simCLR_model
from augment.Find_weighted_neighbor_spots import cal_weighted_near_spots
from augment.Diffusion_model import gene_data_denoising, gene_data_augmentation
from WARGA.train_graph_autoencoders import pretrain_gae
from WARGA.utils import PAGA_trajectory_inference_and_umap_visualization, pseudo_Spatiotemporal_Map, plot_pSM
import matplotlib
matplotlib.use('TkAgg')




def Preprocessing(args):
	args.inputPath = Path(args.basePath)
	args.outPath = Path(args.basePath + '_PearlST/')
	args.outPath.mkdir(parents=True, exist_ok=True)

	##load spatial transcriptomics and histological data
	if args.platform == 'Visium':
		adata = sc.read_visium(args.inputPath)
		adata.var_names_make_unique()
	else:
		adata = sc.read_h5ad(args.inputPath)

	sc.pp.filter_genes(adata, min_cells=5)
	sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=True, inplace=False)
	sc.pp.log1p(adata)
	sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=args.n_top_genes)
	adata = adata[:, adata.var.highly_variable]
	print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.platform == 'Visium':
		adata = st.convert_scanpy(adata)

	"""
	gene expression augmentation
	"""
	count = adata.X.toarray()
	denoised_data = gene_data_denoising(count, iter=3, k=0.2)
	adata.X = denoised_data

	if args.platform == 'Visium':
		# Adding histological image features to adata
		feat = pd.read_csv(args.basePath + '/simCLR_representation_resnet50.csv', index_col=0)
		feat = np.array(feat).astype(np.float32)
		adata.obsm['image_feat'] = feat
		adata.obsm['image_feat_pca'] = feat
		# Find weighted_adjacent_spots
		adata = cal_weighted_near_spots(adata, platform='Visium')

	else:
		cal_weighted_near_spots(adata, platform=args.platform, weights='weights_matrix_nomd')

	adj_spot_index = adata.obsm['near_spots']
	weight_adj = adata.obsm['adjacent_weight']
	augment_data = gene_data_augmentation(count=adata.X, adj_spot_index=adj_spot_index, weight_adj=weight_adj, iter=4, k=0.2)
	adata.obsm['augment_data'] = augment_data
	del adata.obsm['weights_matrix_all']
	del adata.obsm['image_feat']
	del adata.obsm['image_feat_pca']
	del adata.obsm['near_spots']
	del adata.obsm['adjacent_weight']
	return adata



if __name__ == "__main__":
	parser = parameter_setting()
	args = parser.parse_args()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	adata = Preprocessing(args)
	# adding the manual annotations into adata
	ground_truth = pd.read_csv(args.ground_truth_dir + '/' + 'cluster_labels_' + args.dataset_name + '.csv', index_col=0)
	adata.obs['ground_truth'] = list(ground_truth['ground_truth'])

	"""
	learning the low-dimensional embeddings using Wasserstein adversarial regularized graph autoencoder
	"""
	adata, model = pretrain_gae(args, adata)
	torch.save(model.state_dict(), str(args.outPath) + '/model.pt')

	"""
	Plot domains
	"""
	sc.pl.spatial(adata, color='pred_label')
	plt.savefig(str(args.outPath) + '/spatial domains visualization.pdf', dpi=300)

	"""
	UMAP and PAGA trajectory inference visualizations using low-dimensional embeddings
	"""
	PAGA_trajectory_inference_and_umap_visualization(adata, args)

	"""
	pseudo_Spatiotemporal_Map viaualization
	"""
	pseudo_Spatiotemporal_Map(adata, pSM_values_save_filepath=str(args.outPath)+"/pSM_values.tsv", n_neighbors=10, resolution=1.0)
	plot_pSM(adata, pSM_figure_save_filepath=str(args.outPath)+"/pseudo-Spatiotemporal-Map.pdf", colormap='roma',
			 scatter_sz=20., rsz=5., csz=6.5, wspace=.4, hspace=.5, left=0.088, right=1.03, bottom=0.1, top=0.9)
