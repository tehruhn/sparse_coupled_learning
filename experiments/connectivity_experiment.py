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

    sources = 10
    fanouts = [3, 5, 7, 9]

    for fanout in fanouts:
        G, S, H, T = create_low_connectivity_network(sources, fanout)
        # draw_wide_network(G, source_nodes, hidden_layers, target_nodes)
        tri, trt, tei, tet = generate_regression_data_for_experiment()

        linNet = LinearNetwork(G)
        solver = LinearNetworkSolver(linNet)

        K, costs = solver.perform_trial(source_nodes=[0,1],
                                        target_nodes=[T[-2],T[-1]],
                                        ground_nodes=[2],
                                        in_node=tri,
                                        out_node=trt,
                                        lr=0.05,
                                        steps=150000,
                                        debug=True,
                                        every_nth=500,
                                        init_strategy="random"
                                        )
        x, y = zip(*costs)
        y = [a / y[0] for a in y]
        plt.plot(x, y, label=f"Adj Nodes: {fanout}")

    plt.title("Relative Cost vs Iterations for different Fanouts")
    plt.xlabel("Iterations")
    plt.ylabel("Relative Cost")
    plt.yscale('log')
    plt.legend()
    plt.show()

