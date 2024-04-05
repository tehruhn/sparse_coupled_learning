from LinearNetwork import LinearNetwork
from LinearNetworkSolver import LinearNetwork

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

np.random.seed(42)

def create_network(source, hidden, target, num_hidden=1):
    G = nx.DiGraph()
    current_index = 0
    
    # Create source nodes
    source_nodes = []
    for _ in range(source):
        source_nodes.append(current_index)
        current_index += 1
    G.add_nodes_from(source_nodes)

    # Create target nodes
    target_nodes = []
    for _ in range(target):
        target_nodes.append(current_index)
        current_index += 1
    G.add_nodes_from(target_nodes)
    
    # Create hidden layers
    hidden_layers = []
    if num_hidden > 0:
        for layer in range(num_hidden):
            hidden_nodes = []
            for _ in range(hidden):
                hidden_nodes.append(current_index)
                current_index += 1
            hidden_layers.append(hidden_nodes)
            G.add_nodes_from(hidden_nodes)
            
            # Connect nodes
            if layer == 0:
                for source_node in source_nodes:
                    for hidden_node in hidden_nodes:
                        G.add_edge(source_node, hidden_node)
                        G.add_edge(hidden_node, source_node)  # Add reverse connection
            else:
                for prev_hidden_node in hidden_layers[layer-1]:
                    for hidden_node in hidden_nodes:
                        G.add_edge(prev_hidden_node, hidden_node)
                        G.add_edge(hidden_node, prev_hidden_node)  # Add reverse connection
    else:
        # If there are no hidden layers, directly connect source to target nodes
        for source_node in source_nodes:
            for target_node in target_nodes:
                G.add_edge(source_node, target_node)
                G.add_edge(target_node, source_node)  # Add reverse connection
    
    # Connect nodes to target nodes depending on the presence of hidden layers
    for node in hidden_layers[-1] if num_hidden > 0 else source_nodes:
        for target_node in target_nodes:
            G.add_edge(node, target_node)
            G.add_edge(target_node, node)  # Add reverse connection
    
    # Connect source nodes among themselves
    for i in range(1, source):
        G.add_edge(source_nodes[i-1], source_nodes[i])
        G.add_edge(source_nodes[i], source_nodes[i-1])
    
    # Connect target nodes among themselves
    for i in range(1, target):
        G.add_edge(target_nodes[i-1], target_nodes[i])
        G.add_edge(target_nodes[i], target_nodes[i-1])
    
    return G, source_nodes, hidden_layers, target_nodes

def generate_random_regression_data(n_inputs, n_outputs, n_samples=420, train_split=0.95):
    # generate random input pairs
    input_pairs = np.random.uniform(0, 1, (n_samples, n_inputs))
    
    # randomly generate coefficients for a simple linear model
    coefficients = np.random.uniform(0, 0.5, (n_inputs, n_outputs))
    
    # calculate targets based on the generated coefficients
    targets = np.dot(input_pairs, coefficients)
    
    # calculate split index for training and testing data
    split_index = int(n_samples * train_split)
    
    # split data into training and testing sets
    train_inputs = input_pairs[:split_index]
    test_inputs = input_pairs[split_index:]
    
    train_targets = targets[:split_index]
    test_targets = targets[split_index:]
    
    return (train_inputs, train_targets, test_inputs, test_targets)



if __name__ == "__main__":
    source = 3
    hidden = 10
    target = 2
    num_hidden = 1
    G, source_nodes, hidden_layers, target_nodes = create_network(source, hidden, target, num_hidden)
    print("Source Nodes:", source_nodes)
    print("Hidden Layers:", hidden_layers)
    print("Target Nodes:", target_nodes)

    # Drawing the graph
    pos = {}

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