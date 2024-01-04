# -*- coding: utf-8 -*-
"""
Created on Nov 1 2023

@author: Haiyun Wang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse



def parameter_setting():
    parser = argparse.ArgumentParser(description='Spatial transcriptomics analysis')
    BasePath = 'Datasets/151671'
    parser.add_argument('--platform', type=str, default='Visium')
    parser.add_argument('--basePath', '-bp', type=str, default=BasePath,
                        help='base path for the output of 10X pipeline')
    parser.add_argument('--inputPath', '-IP', type=str, default=None, help='data directory')
    parser.add_argument('--tillingPath', '-TP', type=str, default=None, help='image data directory')
    parser.add_argument('--outPath', '-od', type=str, default=None, help='Output path')
    parser.add_argument('--batch_size_I', '-bI', type=int, default=64, help='Batch size for spot image data')
    parser.add_argument('--image_size', '-iS', type=int, default=32, help='image size for spot image data')
    parser.add_argument('--latent_I', '-lI', type=int, default=128,
                        help='Feature dim for latent vector for spot image data')
    parser.add_argument('--max_epoch_I', '-meI', type=int, default=500, help='Max epoches for spot image data')
    parser.add_argument('--current_epoch_I', '-curEI', type=int, default=0, help='current epoches for spot image data')
    parser.add_argument('--lr_I', type=float, default=0.0001, help='Learning rate for spot image data')
    parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true',
                        help=" whether use cuda(default: True)")
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature used in softmax')
    parser.add_argument('--k', type=int, default=200, help='Top k most similar images used to predict the label')
    parser.add_argument('--sizeImage', type=int, default=32, help='Random seed for repeat results')
    parser.add_argument('--test_prop', type=float, default=0.05, help='the proportion data for testing')
    parser.add_argument('--n_top_genes', type=float, default=2000, help='the number of top highly variable genes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--epochs', type=int, default=1270, help='1433 Number of epochs to train')
    parser.add_argument('--hidden1', type=int, default=256, help='Number of units in the first encoding layer')
    parser.add_argument('--hidden2', type=int, default=32, help='Number of units in the second embedding layer')
    parser.add_argument('--reg_in_channels', type=int, default=32,
                        help='Number of units in the input layer of Regularizer')
    parser.add_argument('--reg_hidden1', type=int, default=16,
                        help='Number of units in the first hidden layer of Regularizer')
    parser.add_argument('--reg_hidden2', type=int, default=8,
                        help='Number of units in the second hidden layer of Regularizer')
    parser.add_argument('--gp_lambda', type=float, default=5, help='lambda for gradient penalty')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for Generator')
    parser.add_argument('--reglr', type=float, default=0.0005, help='Initial learning rate for Regularizer')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate (1 - keep probability)')
    parser.add_argument('--dataset_name', type=str, default='151671', help='name of dataset')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--ground_truth_dir', type=str, default='Datasets/SpatialDE_clustering')
    return parser



