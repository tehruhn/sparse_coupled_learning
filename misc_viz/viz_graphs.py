import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G, topology):
    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=10, edge_color='gray')
    plt.title(f"{topology} Graph")
    plt.tight_layout()
    plt.savefig(f"{topology}_graph.png", dpi=300)
    plt.close()

# Generate and visualize different types of graphs
num_nodes = 10

# Random Graph
G_random = nx.gnm_random_graph(num_nodes, num_nodes * 2)
visualize_graph(G_random, "Random")

# Scale-Free Graph
G_scale_free = nx.barabasi_albert_graph(num_nodes, 2)
visualize_graph(G_scale_free, "Scale-Free")

# Small-World Graph
G_small_world = nx.watts_strogatz_graph(num_nodes, 4, 0.1)
visualize_graph(G_small_world, "Small-World")

# Random Regular Graph
G_random_regular = nx.random_regular_graph(3, num_nodes)
visualize_graph(G_random_regular, "Random Regular")

# Ring Graph
G_ring = nx.cycle_graph(num_nodes)
visualize_graph(G_ring, "Ring")

# Complete Graph
G_complete = nx.complete_graph(num_nodes)
visualize_graph(G_complete, "Complete")