import json
import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.graph_utils import create_wide_network
from utils.data_utils import generate_regression_data_for_experiment

from CLSolver.LinearNetwork import LinearNetwork
from CLSolver.LinearNetworkSolver import LinearNetworkSolver

import matplotlib.pyplot as plt

if __name__ == "__main__":

    np.random.seed(42)

    hidden_nodes_range_list =  list(range(1, 11))
    all_costs = {}

    # initialize data generation
    tri, trt, tei, tet = generate_regression_data_for_experiment()

    # for others
    for hidden_nodes in hidden_nodes_range_list:
        print("Trying Hidden Nodes : ", hidden_nodes)
        source, target, num_hidden = 3, 2, 1
        G, S, H, T = create_wide_network(source, hidden_nodes, target, num_hidden)

        linNet = LinearNetwork(G)
        solver = LinearNetworkSolver(linNet)
        
        K, costs = solver.perform_trial(source_nodes=S[0:-1], 
                                        target_nodes=T,
                                        ground_nodes=[S[-1]],
                                        in_node=tri,
                                        out_node=trt,
                                        lr=0.05,
                                        steps=150000,
                                        debug=False,
                                        every_nth=500,
                                        init_strategy="random"
                                        )
        x, y = zip(*costs)
        y = [a/y[0] for a in y]
        plt.plot(x, y, color='blue')
        plt.title("Rel Cost vs Iter")
        plt.xlabel("Iter")
        plt.ylabel("Rel Cost")
        plt.yscale('log')
        plt.show()
