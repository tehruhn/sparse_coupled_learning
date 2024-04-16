import json
import numpy as np
import networkx as nx
from experiment_utils import create_network, generate_random_regression_data
from task_utils import generate_regression_data
from LinearNetwork import LinearNetwork
from LinearNetworkSolver import LinearNetworkSolver

import matplotlib.pyplot as plt

np.random.seed(42)

def create_low_connectivity_network(sources, fanout):
    G = nx.DiGraph()
    hidden = sources + fanout - 1
    last_layer = sources
    curr_index = 0

    source_nodes = []
    for _ in range(sources):
        source_nodes.append(curr_index)
        curr_index += 1
    G.add_nodes_from(source_nodes)

    hidden_nodes = []
    for _ in range(hidden):
        hidden_nodes.append(curr_index)
        curr_index += 1
    G.add_nodes_from(hidden_nodes)

    target_nodes = []
    for _ in range(sources):
        target_nodes.append(curr_index)
        curr_index += 1
    G.add_nodes_from(target_nodes)

    # connect first layer
    start_hidden = 0
    for from_node in source_nodes:
        to_connect = hidden_nodes[start_hidden:start_hidden+fanout]
        for to_node in to_connect:
            G.add_edge(from_node, to_node)
            # print(from_node, to_node)
        start_hidden += 1
    
    # connect second layer
    start_hidden = 0
    for to_node in target_nodes:
        to_connect = hidden_nodes[start_hidden:start_hidden+fanout]
        for from_node in to_connect:
            G.add_edge(from_node, to_node)
            # print(from_node, to_node)
        start_hidden += 1
    
    return G, source_nodes, hidden_nodes, target_nodes

if __name__ == "__main__":
    sources = 5
    fanout = 1
    G, source_nodes, hidden_nodes, target_nodes = create_low_connectivity_network(sources, fanout)
    # Drawing the graph
    pos = {}
    num_hidden = 1
    hidden_layers = [hidden_nodes]

    # Calculate the maximum layer index for positioning
    max_layer_index = num_hidden + 1 if num_hidden > 0 else 1

    # Set positions for source nodes
    for i, node in enumerate(source_nodes):
        pos[node] = (0, i - (len(source_nodes) - 1) / 2)

    # Set positions for hidden layers
    if num_hidden > 0:
        for layer_idx, layer in enumerate(hidden_layers):
            for i, node in enumerate(layer):
                pos[node] = (layer_idx + 1, i - (len(layer) - len(source_nodes)) / 2)

    # Set positions for target nodes
    for i, node in enumerate(target_nodes):
        pos[node] = (max_layer_index, i - (len(target_nodes) - 1) / 2)

    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=9, font_weight='bold', arrowsize=20)
    plt.show()

