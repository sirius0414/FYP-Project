import gc
import logging
from time import time
from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn import cluster
from sklearn import decomposition


rng = RandomState(0)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
n_samples, n_features = faces.shape


# Global centering (focus on one feature, centering all samples)
faces_centered = faces - faces.mean(axis=0)

# Local centering (focus on one sample, centering all features)
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)

# Plot a sample of the input data
n_row, n_col = 5, 5
n_components = n_row * n_col
image_shape = (64, 64)
def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()

# Plot Original faces
plot_gallery("Original Faces", faces[:n_components])

# PCA using randomized SVD with 6 components
pca_estimator = decomposition.PCA(
    n_components=n_components, svd_solver="randomized", whiten=True
)
pca_estimator.fit(faces_centered)
### Top six eigenvetors/ eigenfaces
plot_gallery(
    "Eigenfaces - PCA using randomized SVD", pca_estimator.components_[:n_components]
)
print('the shape of output of 6 eigenfaces is' )
print(pca_estimator.components_[:n_components].shape)
### Dimension Reduction using top 6 eigenvectors on the dataset
print('the shape of dataset is' )
print(faces_centered.shape)
faces_pca = pca_estimator.transform(faces_centered)
print('the shape of output of dimension reduction is' )
print(faces_pca.shape)
### Reconstruction using top 6 eigenvectors
faces_reconstructed = pca_estimator.inverse_transform(faces_pca)
print('the shape of output of reconstruction is' )
print(faces_reconstructed.shape)
### Plot reconstructed faces
plot_gallery("Reconstruction", faces_reconstructed[:n_components])

# Eigenvector matrix
eigenvector_matrix = pca_estimator.components_
# Transpose of eigenvector matrix
eigenvector_matrix_T = np.transpose(eigenvector_matrix)
# Product of eigenvector matrix and its transpose
eigenvector_matrix_product = np.dot(eigenvector_matrix, eigenvector_matrix_T)
# Shape of the product
print('the shape of product of eigenvector matrix and its transpose is' )
print(eigenvector_matrix_product.shape)
# If the product is an identity matrix, then the eigenvectors are orthogonal
print('the product of eigenvector matrix and its transpose is')
print(eigenvector_matrix_product)

