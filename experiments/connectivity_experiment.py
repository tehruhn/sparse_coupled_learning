import json
import os
import sys
import numpy as np
import networkx as nx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.graph_utils import create_low_connectivity_network, draw_wide_network
from utils.data_utils import generate_regression_data_for_experiment

from CLSolver.LinearNetwork import LinearNetwork
from CLSolver.LinearNetworkSolver import LinearNetworkSolver

import matplotlib.pyplot as plt


if __name__ == "__main__":

    np.random.seed(42)

    sources = 5
    fanout = 1
    G, source_nodes, hidden_layers, target_nodes = create_low_connectivity_network(sources, fanout)
    # draw_wide_network(G, source_nodes, hidden_layers, target_nodes)
    

