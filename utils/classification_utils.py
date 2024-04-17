import json
import os
import sys
import numpy as np
import networkx as nx
from sklearn import datasets

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.graph_utils import create_low_connectivity_network, draw_wide_network
from utils.data_utils import generate_regression_data_for_experiment

from CLSolver.LinearNetwork import LinearNetwork
from CLSolver.LinearNetworkSolver import LinearNetworkSolver

def extract_linearly_separable_points(features, target_classes):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_subset = X[:, features]

    X_subset = X_subset[np.isin(y, target_classes)]
    y_subset = y[np.isin(y, target_classes)]

    y_subset[y_subset == target_classes[0]] = -1
    y_subset[y_subset == target_classes[1]] = 1

    return X_subset, y_subset

if __name__ == "__main__":
    features = [0, 1, 2, 3]
    target_classes = [0, 1]
    X_subset, y_subset = extract_linearly_separable_points(features, target_classes)

    print("Number of samples in the first class:", np.sum(y_subset == -1))
    print("Number of samples in the second class:", np.sum(y_subset == 1))

    print("Subset of the dataset:")
    print(X_subset[:5])
    print(y_subset[:5])
