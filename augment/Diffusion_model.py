# -*- coding: utf-8 -*-
"""
Created on Nov 1 2023

@author: Haiyun Wang
"""

import numpy as np
import math


def coefficient(I, k):
    """
    :param I:
    :param k:
    :return:
    """
    return 1 / (1 + ((I / k) ** 2))


def anisotropic_diffusion(count, adj_spot_index, weight_adj, image_loc, iterations, k, lamb=0.1):
    """
    :param image_location:
    :param log_freq:
    :param iterations:
    :param k:
    :param lamb:
    :return:
    """
    image = count[image_loc]
    new_image = np.zeros(image.shape, dtype=image.dtype)

    result = []

    for t in range(iterations):
        # unweighted diffusion

        I_North = count[adj_spot_index[image_loc][0]] - image
        I_South = count[adj_spot_index[image_loc][1]] - image
        I_East = count[adj_spot_index[image_loc][2]] - image
        I_West = count[adj_spot_index[image_loc][3]] - image

        new_image = image + lamb * (
                coefficient(I_North, k) * I_North +
                coefficient(I_South, k) * I_South +
                coefficient(I_East, k) * I_East +
                coefficient(I_West, k) * I_West
        )
        image = new_image

        if (t+1) == iterations:
            result.append(image.copy())

    return result


def anisotropic_denoising(count, image_loc, iterations, k, lamb=0.1):
    """
    :param image_location:
    :param log_freq:
    :param iterations:
    :param k:
    :param lamb:
    :return:
    """
    image = count[image_loc]
    new_image = np.zeros(image.shape, dtype=image.dtype)

    result = []

    for t in range(iterations):

        I_North = image[:-2, 1:-1] - image[1:-1, 1:-1]
        I_South = image[2:, 1:-1] - image[1:-1, 1:-1]
        I_East = image[1:-1, 2:] - image[1:-1, 1:-1]
        I_West = image[1:-1, :-2] - image[1:-1, 1:-1]

        new_image[1:-1, 1:-1] = image[1:-1, 1:-1] + lamb * (
            coefficient(I_North, k) * I_North +
            coefficient(I_South, k) * I_South +
            coefficient(I_East, k) * I_East +
            coefficient(I_West, k) * I_West
        )
        image = new_image
        if (t+1) == iterations:
            result.append(image.copy())

    return result


def gene_data_denoising(count, iter, k):
    """
    :param count: gene expression matrix
    :param iter: the number of iterations
    :param k: hyperparameter default by 0.2
    :return: denoised gene expression matrix
    """
    count = count.reshape(len(count), 50, 40)
    iterations = iter
    k = k
    lamb = 0.1
    augment_data = count.copy()

    for i in range(len(count)):
        loc = i
        PDE_image = anisotropic_denoising(count=count,
                                          image_loc=loc, iterations=iterations, k=k, lamb=lamb)
        augment_data[i] = PDE_image[0]

    augment_data = augment_data.reshape(len(count), 2000)
    return augment_data


def gene_data_augmentation(count, adj_spot_index, weight_adj, iter, k):
    """
    :param loc: spot index
    :param count: gene expression matrix
    :param adj_spot_index: index of neighbor spots for each target spot
    :param weight_adj: the weight of neighboring spots
    :param k: hyperparameter default by 0.2
    :return: augment gene expression matrix using gene expression from neighbors
    """
    iterations = iter
    k = k
    lamb = 0.1
    augment_data = count.copy()

    for i in range(len(count)):
        loc = i
        PDE_image = anisotropic_diffusion(count=count, adj_spot_index=adj_spot_index, weight_adj=weight_adj,
                                          image_loc=loc, iterations=iterations, k=k, lamb=lamb)
        augment_data[i] = PDE_image[0]

    return augment_data


