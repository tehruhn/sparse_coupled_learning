import json
import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.graph_utils import create_topology_network
from utils.data_utils import generate_regression_data_for_experiment

from CLSolver.LinearNetwork import LinearNetwork
from CLSolver.LinearNetworkSolver import LinearNetworkSolver

import matplotlib.pyplot as plt

def run_solver(G, tri, trt):
    linNet = LinearNetwork(G)
    solver = LinearNetworkSolver(linNet)
    
    K, costs = solver.perform_trial(
        source_nodes=[0, 1], 
        target_nodes=[num_nodes-2, num_nodes-1], 
        ground_nodes=[2], 
        in_node=tri, 
        out_node=trt, 
        lr=0.05, 
        steps=100, 
        debug=True,
        every_nth=5000)
    
    return costs[-1][1]  # Return the final cost value

if __name__ == "__main__":

    # generate random regression data
    tri, trt, tei, tet = generate_regression_data_for_experiment()

    # define the number of nodes and the topologies to test
    num_nodes = 100
    topologies = ['random', 'scale_free', 'small_world', 'random_regular', 'ring', 'complete']

    final_costs = {}

    for topology in topologies:
        print(f"Testing topology: {topology}")
        final_costs[topology] = []
        
        for _ in range(10):  # run 10 trials for each topology
            G = create_topology_network(num_nodes, topology)
            final_cost = run_solver(G, tri, trt)
            final_costs[topology].append(final_cost)

    # plot the final costs for each topology
    plt.figure(figsize=(10, 6))
    plt.boxplot([final_costs[topology] for topology in topologies], labels=topologies)
    plt.xlabel('Topology')
    plt.ylabel('Final Cost')
    plt.title('Final Cost vs Topology')
    plt.grid(True)
    plt.show()