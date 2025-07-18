# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scikit-learn models with OpenVINO optimizations for OTX."""

from .classification.logistic_regression import LogisticRegression
from .classification.random_forest import RandomForestClassifier
from .classification.knn import KNeighborsClassifier
from .classification.svc import SVC
from .classification.nusvc import NuSVC
from .regression.linear_regression import LinearRegression
from .regression.ridge import Ridge
from .regression.lasso import Lasso
from .regression.elastic_net import ElasticNet  
from .regression.random_forest_regressor import RandomForestRegressor
from .regression.svr import SVR
from .regression.nusvr import NuSVR
from .clustering.kmeans import KMeans
from .clustering.dbscan import DBSCAN
from .decomposition.pca import PCA
from .decomposition.tsne import TSNE
from .neighbors.nearest_neighbors import NearestNeighbors

__all__ = [
    # Classification
    "LogisticRegression",
    "RandomForestClassifier",
    "KNeighborsClassifier",
    "SVC",
    "NuSVC",
    # Regression
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "RandomForestRegressor",
    "SVR",
    "NuSVR",
    "KNeighborsRegressor",
    # Clustering
    "KMeans",
    "DBSCAN",
    # Decomposition
    "PCA",
    "IncrementalPCA",
    "TSNE",
    # Neighbors
    "NearestNeighbors",
]
