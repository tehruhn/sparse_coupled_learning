import json
import sklearn
import os
import sys
import numpy as np
import networkx as nx
from sklearn import datasets
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.graph_utils import create_random_network
from utils.data_utils import generate_regression_data_for_experiment
from utils.classification_utils import extract_linearly_separable_points

from CLSolver.LinearNetwork import LinearNetwork
from CLSolver.LinearNetworkSolver import LinearNetworkSolver

if __name__ == "__main__":
    np.random.seed(42)
    
    features = [0, 1, 2, 3]
    target_classes = [0, 1]
    X_subset, y_subset = extract_linearly_separable_points(features, target_classes)

    sources = len(features)
    nodes = list(range(5*sources))
    num_nodes = 5*sources
    num_edges = 25*sources
    G = create_random_network(num_nodes, num_edges)
    # nx.draw(G)
    # plt.show()
    print(X_subset.shape, y_subset.shape)

    linNet = LinearNetwork(G)
    solver = LinearNetworkSolver(linNet)

    # print(nodes[:sources], nodes[-sources:],nodes[sources+1])

    K, costs = solver.perform_trial(source_nodes=nodes[:sources], 
                                    target_nodes=nodes[-sources:],
                                    ground_nodes=[nodes[sources+1]],
                                    in_node=X_subset,
                                    out_node=X_subset,
                                    lr=0.05, 
                                    steps=2, 
                                    debug=True,
                                    every_nth=100,
                                    init_strategy="random")
    
    print(solver.PFs.shape)
