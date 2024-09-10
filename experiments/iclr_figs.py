import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def create_simple_network():
    G = nx.Graph()
    G.add_edges_from([(1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (4,5)])
    return G

def draw_network(ax, G, pos, edge_weights, title, source_node=None, target_node=None):
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    for (u, v, w) in G.edges(data='weight'):
        ax.annotate("",
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="<->", color="0.5",
                                    shrinkA=10, shrinkB=10,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3*w)
                                    ),
                                    linewidth=1+5*w))
    
    if source_node:
        nx.draw_networkx_nodes(G, pos, nodelist=[source_node], node_color='green', node_size=500, ax=ax)
    if target_node:
        nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_color='red', node_size=500, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')

def draw_learning_rule(ax):
    ax.text(0.5, 0.6, r"$\frac{dk_{ij}}{dt} = \gamma ((v_{F,i} - v_{F,j})^2 - (v_{C,i} - v_{C,j})^2)$", 
            horizontalalignment='center', verticalalignment='center', fontsize=16)
    ax.text(0.5, 0.3, "Contrastive Hebbian Learning Rule", horizontalalignment='center', fontsize=12)
    ax.axis('off')


# Create the network
G = create_simple_network()
pos = nx.spring_layout(G, k=1, iterations=50)

# Create and save individual plots
fig_size = (5, 5)
dpi = 300  # Set a high DPI for better quality

# A) Initial Network State
fig, ax = plt.subplots(figsize=fig_size)
initial_weights = {(u,v): 0.5 for (u,v) in G.edges()}
nx.set_edge_attributes(G, initial_weights, 'weight')
draw_network(ax, G, pos, initial_weights, "Initial Network State", source_node=1, target_node=5)
plt.savefig('initial_state.png', format='png', dpi=dpi, bbox_inches='tight')
plt.close()

# B) Network After Learning
fig, ax = plt.subplots(figsize=fig_size)
final_weights = {(u,v): 0.5 + 0.5*np.random.rand() for (u,v) in G.edges()}
nx.set_edge_attributes(G, final_weights, 'weight')
draw_network(ax, G, pos, final_weights, "Network After Learning", source_node=1, target_node=5)
plt.savefig('after_learning.png', format='png', dpi=dpi, bbox_inches='tight')
plt.close()

# C) Free State
fig, ax = plt.subplots(figsize=fig_size)
free_weights = {(u,v): 0.3 + 0.4*np.random.rand() for (u,v) in G.edges()}
nx.set_edge_attributes(G, free_weights, 'weight')
draw_network(ax, G, pos, free_weights, "Free State", source_node=1)
plt.savefig('free_state.png', format='png', dpi=dpi, bbox_inches='tight')
plt.close()

# D) Clamped State
fig, ax = plt.subplots(figsize=fig_size)
clamped_weights = {(u,v): 0.3 + 0.4*np.random.rand() for (u,v) in G.edges()}
nx.set_edge_attributes(G, clamped_weights, 'weight')
draw_network(ax, G, pos, clamped_weights, "Clamped State", source_node=1, target_node=5)
plt.savefig('clamped_state.png', format='png', dpi=dpi, bbox_inches='tight')
plt.close()

print("All plots have been saved as individual PNG files.")