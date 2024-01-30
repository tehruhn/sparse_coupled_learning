import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import spsolve

# Be careful about this since Laplacian can be zero if there are orphan nodes
def create_connected_random_graph(N, additional_edge_probability):
    edges = set((i, (i + 1) % N) for i in range(N))
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < additional_edge_probability:
                edge = (i, j)
                if edge not in edges:
                    edges.add(edge)
    
    return list(edges)

def construct_flow_laplacian(N, edges, conductances):
    row_indices = []
    col_indices = []
    data = []
    for (i, j), conductance in zip(edges, conductances):
        row_indices.extend([i, j])
        col_indices.extend([j, i])
        data.extend([conductance, conductance])
    adjacency_matrix_csr = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    L_csr = laplacian(adjacency_matrix_csr, normed=False)
    return L_csr

def solve_for_state_pressures(L, external_flows, fixed_nodes, fixed_pressures):
    L_lil = L.tolil()
    for fixed_node, pressure in zip(fixed_nodes, fixed_pressures):
        L_lil[fixed_node, :] = 0
        L_lil[:, fixed_node] = 0
        L_lil[fixed_node, fixed_node] = 1
        external_flows[fixed_node] = pressure
    L_csr = L_lil.tocsr()
    pressures = spsolve(L_csr, external_flows)
    return pressures

def solve_for_free_state_pressures(L, N, source_nodes, source_pressures):
    external_flows = np.zeros(N)
    return solve_for_state_pressures(L.copy(), external_flows, source_nodes, source_pressures)

def solve_for_clamped_state_pressures(L, N, source_nodes, source_pressures, target_nodes, target_pressures):
    external_flows = np.zeros(N)
    all_fixed_nodes = source_nodes + target_nodes
    all_fixed_pressures = source_pressures + target_pressures
    return solve_for_state_pressures(L.copy(), external_flows, all_fixed_nodes, all_fixed_pressures)

def calculate_dissipated_power(edges, pressures, conductances):
    power = 0
    for (i, j), k in zip(edges, conductances):
        delta_p = pressures[i] - pressures[j]
        power += 0.5 * k * delta_p**2
    return power

def apply_learning_rule(edges, pressures_free, pressures_clamped, conductances, alpha, eta):
    new_conductances = conductances.copy()
    for idx, (i, j) in enumerate(edges):
        delta_p_free = pressures_free[i] - pressures_free[j]
        delta_p_clamped = pressures_clamped[i] - pressures_clamped[j]
        new_conductances[idx] += alpha * eta**(-1) * (delta_p_free**2 - delta_p_clamped**2)
    return new_conductances

def learning_process(N, edges, conductances, source_nodes, source_pressures, target_nodes, target_pressures, alpha, eta, epochs):
    costs = []
    L = construct_flow_laplacian(N, edges, conductances)
    for epoch in range(epochs):
        pressures_free = solve_for_free_state_pressures(L, N, source_nodes, source_pressures)
        clamped_target_pressures = [pF + eta * (pT - pF) for pF, pT in zip(pressures_free, target_pressures)]
        pressures_clamped = solve_for_clamped_state_pressures(L, N, source_nodes, source_pressures, target_nodes, clamped_target_pressures)
        conductances = apply_learning_rule(edges, pressures_free, pressures_clamped, conductances, alpha, eta)
        L = construct_flow_laplacian(N, edges, conductances)
        cost = calculate_dissipated_power(edges, pressures_free, conductances)
        costs.append(cost)
    return costs

def plot_flow_network(N, edges, conductances, pressures, fixed_nodes, layout="random"):
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, pressure=pressures[i])
    for (i, j), conductance in zip(edges, conductances):
        G.add_edge(i, j, weight=conductance)
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'planar':
        pos = nx.planar_layout(G)
    elif layout == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(G)
    else:
        pos = nx.spring_layout(G)
    node_sizes = [300 * (np.abs(pressure) + 1) for pressure in pressures]
    edge_widths = [5 * conductance for conductance in conductances]
    nx.draw(G, pos, node_size=node_sizes, width=edge_widths, with_labels=True)
    fixed_node_colors = ['red' if node in fixed_nodes else 'blue' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=fixed_node_colors)

    plt.title("Flow Network Visualization")
    plt.grid(True)
    plt.show()