import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca(data, n_components=None, scale_data=True, plot=True):
    """
    Performs PCA on a given dataset and plots a scree plot.
    
    Parameters:
    - data: pandas DataFrame or 2D numpy array
    - n_components: Number of components to keep (default is all)
    - scale_data: Whether to standardize the data before PCA
    - plot: Whether to plot a scree plot
    
    Returns:
    - pca: Fitted PCA object
    - principal_components: Transformed data
    - explained_variance_ratio: Explained variance ratio for each component
    """
    if isinstance(data, pd.DataFrame):
        features = data.values
    else:
        features = data

    # Standardize data
    if scale_data:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)
    explained_variance_ratio = pca.explained_variance_ratio_

    # Scree plot
    if plot:
        sns.set(style='whitegrid')
        plt.figure(figsize=(10, 6))
        components = np.arange(1, len(explained_variance_ratio) + 1)
        sns.barplot(x=components, y=explained_variance_ratio, palette="viridis")
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.xticks(components)
        plt.tight_layout()
        plt.show()

    return pca, principal_components, explained_variance_ratio


def perform_pca_from_scratch(data, n_components=None, scale_data=True, plot=True):
    """
    Performs PCA without using scikit-learn.
    
    Parameters:
    - data: pandas DataFrame or 2D numpy array
    - n_components: Number of principal components to return (default is all)
    - scale_data: Standardize features before PCA
    - plot: Whether to show a scree plot
    
    Returns:
    - components: Principal components (eigenvectors)
    - explained_variance_ratio: Variance ratio of each PC
    - transformed_data: Data projected onto principal components
    """
    # Convert to NumPy array
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Standardize data
    if scale_data:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        X = (data - mean) / std
    else:
        X = data

    # Covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Explained variance ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    # Select number of components
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]

    # Project data
    transformed_data = np.dot(X, eigenvectors)

    # Scree plot
    if plot:
        sns.set(style='whitegrid')
        plt.figure(figsize=(10, 6))
        components = np.arange(1, len(explained_variance_ratio) + 1)
        sns.barplot(x=components, y=explained_variance_ratio, palette='mako')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot (From Scratch)')
        plt.xticks(components)
        plt.tight_layout()
        plt.show()

    return eigenvectors, explained_variance_ratio, transformed_data
