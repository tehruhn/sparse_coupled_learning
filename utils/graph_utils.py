import networkx as nx
import matplotlib.pyplot as plt

def create_wide_network(source, hidden, target, num_hidden=1):
    """
    Creates a wide network with a given number of hidden layers.
    """
    G = nx.DiGraph()
    current_index = 0
    
    # create source nodes
    source_nodes = []
    for _ in range(source):
        source_nodes.append(current_index)
        current_index += 1
    G.add_nodes_from(source_nodes)

    # create target nodes
    target_nodes = []
    for _ in range(target):
        target_nodes.append(current_index)
        current_index += 1
    G.add_nodes_from(target_nodes)
    
    # create hidden layers
    hidden_layers = []
    for layer in range(num_hidden):
        hidden_nodes = []
        for _ in range(hidden):
            hidden_nodes.append(current_index)
            current_index += 1
        hidden_layers.append(hidden_nodes)
        G.add_nodes_from(hidden_nodes)
        
        # connect nodes
        if layer == 0:
            for source_node in source_nodes:
                for hidden_node in hidden_nodes:
                    G.add_edge(source_node, hidden_node)
        else:
            for prev_hidden_node in hidden_layers[layer-1]:
                for hidden_node in hidden_nodes:
                    G.add_edge(prev_hidden_node, hidden_node)

    # connect nodes to target nodes depending on the presence of hidden layers
    for node in hidden_layers[-1] if num_hidden > 0 else source_nodes:
        for target_node in target_nodes:
            G.add_edge(node, target_node)
            # G.add_edge(target_node, node)  # Add reverse connection
    
    return G, source_nodes, hidden_layers, target_nodes

def create_low_connectivity_network(sources, fanout):
    """
    Create a low connectivity network with a certain 
    fanout. Has the same config as a single layer wide network.
    """
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
    
    return G, source_nodes, [hidden_nodes], target_nodes


def create_random_network(num_nodes, num_edges):
    """
    Create a random network with given number of nodes and edges
    """
    G = nx.gnm_random_graph(num_nodes, num_edges)
    return G

def create_topology_network(num_nodes, topology):
    """
    Create a network with given number of nodes and of a 
    certain type of topology.
    """
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
    
    # ensure the graph is connected
    if not nx.is_connected(G):
        components = nx.connected_components(G)
        largest_component = max(components, key=len)
        G = G.subgraph(largest_component).copy()
    
    return G

def draw_wide_network(G, source_nodes, hidden_layers, target_nodes):
    """
    Draws a wide network graph
    """
    # drawing the graph
    pos = {}
    num_hidden = len(hidden_layers)
    # calculate the maximum layer index for positioning
    max_layer_index = num_hidden + 1 if num_hidden > 0 else 1

    # set positions for source nodes
    for i, node in enumerate(source_nodes):
        pos[node] = (0, i - (len(source_nodes) - 1) / 2)

    # set positions for hidden layers
    if num_hidden > 0:
        for layer_idx, layer in enumerate(hidden_layers):
            for i, node in enumerate(layer):
                pos[node] = (layer_idx + 1, i - (len(layer) - len(source_nodes)) / 2)

    # set positions for target nodes
    for i, node in enumerate(target_nodes):
        pos[node] = (max_layer_index, i - (len(target_nodes) - 1) / 2)

    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=9, font_weight='bold', arrowsize=20)
    plt.show()