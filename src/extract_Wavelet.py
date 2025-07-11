# -*- coding: utf-8 -*-
"""
Script for extracting the Wavlet details of an image and construct a new
dataset where each image is composed of the three channels:
    (cH, cV, cD)

where cH, cV, and cD are the horizzontal, vertical, and diagonal details
of the wavelet decomposition.
This is used to implement the fusion strategies used in analysing building
images for the damage level classification on a dataset labelled from scratch
as described in:

- Simone Saquella, Michele Scarpiniti, Wangyi Pu, Livio Pedone, Giulia Angelucci,
Michele Matteoni, Mattia Francioli, Stefano Pampanin, "Post-earthquake Damage
Assessment of Buildings Exploiting Data Fusion", in 2025 International Joint
Conference on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import pywt
import cv2



# Convert and RGB image to a gray-scale one -----------------------------------
def rgb2gray(img):
    gray = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]

    return gray


# Scale matrix in range [0, 1] ------------------------------------------------
def scale(matrix):
    # Perform min-max scaling
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    scaled_matrix = (matrix - min_val) / (max_val - min_val)

    return scaled_matrix




# Set data folder
data_folder = '../Data/'

training_set = data_folder + 'X_train.npy'
testing_set  = data_folder + 'X_test.npy'



# %% Training set

X = np.load(training_set)

L = X.shape[0]
N = X.shape[1]
M = X.shape[2]
W = np.empty([L, N, M, 3])


for i in range(L):
    img = rgb2gray(X[i])
    cA, (cH, cV, cD) = pywt.dwt2(np.round(img), 'haar')

    r = np.round(255*scale(cH))
    g = np.round(255*scale(cV))
    b = np.round(255*scale(cD))

    Z = np.stack([r, g, b], axis=-1)

    W[i,:,:,:] = cv2.resize(Z, (N, M), interpolation=cv2.INTER_NEAREST)


W = np.array(W, dtype = 'uint8')
np.save('X_train_wavelet.npy', W)



# %% Test set

X = np.load(testing_set)

L = X.shape[0]
N = X.shape[1]
M = X.shape[2]
W = np.empty([L, N, M, 3])


for i in range(L):
    img = rgb2gray(X[i])
    cA, (cH, cV, cD) = pywt.dwt2(np.round(img), 'haar')

    r = np.round(255*scale(cH))
    g = np.round(255*scale(cV))
    b = np.round(255*scale(cD))

    Z = np.stack([r, g, b], axis=-1)

    W[i,:,:,:] = cv2.resize(Z, (N, M), interpolation=cv2.INTER_NEAREST)


W = np.array(W, dtype = 'uint8')
np.save('X_wavelet_test.npy', W)
