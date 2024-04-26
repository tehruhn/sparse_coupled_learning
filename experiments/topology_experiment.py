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
    num_nodes = G.number_of_nodes()
    print(G.nodes())
    K, costs = solver.perform_trial(
        source_nodes=[0, 1],
        target_nodes=[num_nodes-2, num_nodes-1],
        ground_nodes=[2],
        in_node=tri,
        out_node=trt,
        lr=0.05,
        steps=25000,
        debug=True,
        every_nth=5000
    )
    return costs[-1][1]  # Return the final cost value

if __name__ == "__main__":
    # generate random regression data
    tri, trt, tei, tet = generate_regression_data_for_experiment()

    # define the number of nodes and the topologies to test
    num_edges_list = [50, 100, 250, 500, 1000]
    topologies = [
        'random',
        'scale_free',
        'small_world',
        'random_regular',
        'ring',
        'complete',
        'square_lattice',
        'triangular_lattice'
    ]

    final_costs = {topology: [[] for _ in range(len(num_edges_list))] for topology in topologies}

    for i, num_edges in enumerate(num_edges_list):
        for topology in topologies:
            print(f"Testing topology: {topology} with {num_edges} edges")
            for _ in range(10):  # run 10 trials for each topology
                G = create_topology_network(num_edges, topology)
                final_cost = run_solver(G, tri, trt)
                final_costs[topology][i].append(final_cost)

    # plot the final costs for each topology
    plt.figure(figsize=(10, 6))
    for topology in topologies:
        inv_num_edges = 1 / np.array(num_edges_list)
        mean_costs = np.mean(final_costs[topology], axis=1)
        std_costs = np.std(final_costs[topology], axis=1)
        plt.errorbar(inv_num_edges, mean_costs, yerr=std_costs, capsize=5, label=topology)

    plt.xlabel('1/N (Inverse Number of Edges)')
    plt.ylabel('Final Cost')
    plt.yscale('log')
    plt.title('Final Cost vs 1/N for Different Topologies')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./inv_edge_plot.png")

    for topology in topologies:
        mean_costs = np.mean(final_costs[topology], axis=1)
        std_costs = np.std(final_costs[topology], axis=1)
        print(f"{topology} & {' & '.join(f'{mean:.4f} ({std:.4f})' for mean, std in zip(mean_costs, std_costs))}")

# # topology vs size

# random & 0.0000 (0.0000) & 0.0002 (0.0003) & 0.0010 (0.0008) & 0.0014 (0.0009) & 0.0022 (0.0009)
# scale_free & 0.0037 (0.0004) & 0.0000 (0.0000) & 0.0047 (0.0017) & 0.0025 (0.0006) & 0.0080 (0.0059)
# small_world & 0.0000 (0.0000) & 0.0405 (0.0475) & 0.0002 (0.0001) & 0.1136 (0.0001) & 0.1136 (0.0001)
# random_regular & 0.0000 (0.0000) & 0.0000 (0.0000) & 0.0001 (0.0001) & 0.0000 (0.0000) & 0.0000 (0.0000)
# ring & 0.0564 (0.0076) & 0.0549 (0.0037) & 0.1042 (0.0562) & 0.1817 (0.0733) & 0.2082 (0.0639)
# complete & 0.0000 (0.0000) & 0.0000 (0.0000) & 0.0000 (0.0000) & 0.0000 (0.0000) & 0.0001 (0.0000)
# square_lattice & 0.0064 (0.0017) & 0.0113 (0.0031) & 0.0686 (0.0364) & 0.2510 (0.0002) & 0.2447 (0.0011)
# triangular_lattice & 0.0007 (0.0009) & 0.0000 (0.0000) & 0.1134 (0.0002) & 0.0000 (0.0000) & 0.1134 (0.0001)