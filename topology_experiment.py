import json
import numpy as np
import networkx as nx
from task_utils import generate_regression_data
from LinearNetwork import LinearNetwork
from LinearNetworkSolver import LinearNetworkSolver
import matplotlib.pyplot as plt

def create_network(num_nodes, topology):
    if topology == 'random':
        G = nx.gnm_random_graph(num_nodes, num_nodes * 5)
    elif topology == 'scale_free':
        G = nx.barabasi_albert_graph(num_nodes, 2)
    elif topology == 'small_world':
        G = nx.watts_strogatz_graph(num_nodes, 4, 0.1)
    elif topology == 'random_regular':
        G = nx.random_regular_graph(4, num_nodes)
    elif topology == 'ring':
        G = nx.cycle_graph(num_nodes)
    elif topology == 'complete':
        G = nx.complete_graph(num_nodes)
    else:
        raise ValueError(f"Invalid topology: {topology}")
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        components = nx.connected_components(G)
        largest_component = max(components, key=len)
        G = G.subgraph(largest_component).copy()
    
    return G

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

# Generate random regression data
tri, trt, tei, tet = generate_regression_data()

# Define the number of nodes and the topologies to test
num_nodes = 100
topologies = ['random', 'scale_free', 'small_world', 'random_regular', 'ring', 'complete']

final_costs = {}

for topology in topologies:
    print(f"Testing topology: {topology}")
    final_costs[topology] = []
    
    for _ in range(10):  # Run 10 trials for each topology
        G = create_network(num_nodes, topology)
        final_cost = run_solver(G, tri, trt)
        final_costs[topology].append(final_cost)

# Plot the final costs for each topology
plt.figure(figsize=(10, 6))
plt.boxplot([final_costs[topology] for topology in topologies], labels=topologies)
plt.xlabel('Topology')
plt.ylabel('Final Cost')
plt.title('Final Cost vs Topology')
plt.grid(True)
plt.show()