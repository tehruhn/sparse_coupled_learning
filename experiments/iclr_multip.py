import os
import sys
import numpy as np
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.data_utils import generate_regression_data_for_experiment
from CLSolver.LinearNetwork import LinearNetwork
from CLSolver.LinearNetworkSolver import LinearNetworkSolver

def create_topology_network(target_edges, topology):
    """
    Create a network with approximately the target number of edges and of a certain type of topology.
    """
    if topology == '2d_modular':
        module_size = max(2, int(np.sqrt(target_edges / 16)))
        num_modules = 4
        G = nx.Graph()
        
        for i in range(num_modules):
            module = nx.complete_graph(module_size**2)
            module = nx.relabel_nodes(module, lambda x: x + i*module_size**2)
            G = nx.compose(G, module)
        
        for i in range(2):
            for j in range(2):
                current_module = i * 2 + j
                right_module = i * 2 + (j + 1) % 2
                down_module = ((i + 1) % 2) * 2 + j
                
                for k in range(module_size):
                    G.add_edge(current_module*module_size**2 + k*module_size + (module_size-1),
                               right_module*module_size**2 + k*module_size)
                    G.add_edge(current_module*module_size**2 + (module_size-1)*module_size + k,
                               down_module*module_size**2 + k)

    elif topology == 'small_world':
        num_nodes = int(np.sqrt(target_edges * 2))
        k = 4
        p = 0.1
        G = nx.watts_strogatz_graph(num_nodes, k, p)

    elif topology == 'square_lattice':
        side_length = int(np.sqrt(target_edges / 2))
        G = nx.grid_2d_graph(side_length, side_length)
        G = nx.convert_node_labels_to_integers(G)

    elif topology == 'layer':
        num_layers = 4
        nodes_per_layer = max(2, int(np.sqrt(target_edges / 3)))
        G = nx.DiGraph()
        
        for i in range(num_layers * nodes_per_layer):
            G.add_node(i)
        
        for layer in range(num_layers - 1):
            for node in range(nodes_per_layer):
                for next_node in range(nodes_per_layer):
                    G.add_edge(layer * nodes_per_layer + node, 
                               (layer + 1) * nodes_per_layer + next_node)

    else:
        raise ValueError(f"Invalid topology: {topology}")
    
    return G

def visualize_and_save_topology(topology, target_edges, filename):
    G = create_topology_network(target_edges, topology)
    plt.figure(figsize=(12, 12))
    
    if topology == 'square_lattice':
        side_length = int(np.sqrt(G.number_of_nodes()))
        pos = {node: (node % side_length, side_length - 1 - node // side_length) for node in G.nodes()}
    elif topology == '2d_modular':
        module_size = int(np.sqrt(G.number_of_nodes() // 4))
        pos = {}
        for node in G.nodes():
            module = node // (module_size ** 2)
            within_module = node % (module_size ** 2)
            module_x = module % 2
            module_y = module // 2
            x = module_x * (module_size + 1) + (within_module % module_size)
            y = module_y * (module_size + 1) + (within_module // module_size)
            pos[node] = (x, -y)
        
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)
        label_pos = {k: (v[0], v[1]-0.1) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, font_size=6)
        
    elif topology == 'layer':
        num_layers = 4
        nodes_per_layer = G.number_of_nodes() // num_layers
        pos = {}
        for node in G.nodes():
            layer = node // nodes_per_layer
            within_layer = node % nodes_per_layer
            x = layer
            y = within_layer - nodes_per_layer / 2
            pos[node] = (x, y)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    if topology != '2d_modular':
        nx.draw(G, pos, node_color='lightblue', node_size=300, with_labels=True, font_size=8, font_weight='bold', arrows=True)
    
    plt.title(f"{topology.replace('_', ' ').title()} Topology ({G.number_of_edges()} edges)", fontsize=30)  # Increased font size by 3x
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# def visualize_and_save_topology(topology, target_edges, filename):
#     G = create_topology_network(target_edges, topology)
#     plt.figure(figsize=(12, 12))
    
#     if topology == 'square_lattice':
#         side_length = int(np.sqrt(G.number_of_nodes()))
#         pos = {node: (node % side_length, side_length - 1 - node // side_length) for node in G.nodes()}
#     elif topology == '2d_modular':
#         module_size = int(np.sqrt(G.number_of_nodes() // 4))
#         pos = {}
#         for node in G.nodes():
#             module = node // (module_size ** 2)
#             within_module = node % (module_size ** 2)
#             module_x = module % 2
#             module_y = module // 2
#             x = module_x * (module_size + 1) + (within_module % module_size)
#             y = module_y * (module_size + 1) + (within_module // module_size)
#             pos[node] = (x, -y)
        
#         nx.draw_networkx_edges(G, pos, alpha=0.2)
#         nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)
#         label_pos = {k: (v[0], v[1]-0.1) for k, v in pos.items()}
#         nx.draw_networkx_labels(G, label_pos, font_size=6)
        
#     elif topology == 'layer':
#         num_layers = 4
#         nodes_per_layer = G.number_of_nodes() // num_layers
#         pos = {}
#         for node in G.nodes():
#             layer = node // nodes_per_layer
#             within_layer = node % nodes_per_layer
#             x = layer
#             y = within_layer - nodes_per_layer / 2
#             pos[node] = (x, y)
#     else:
#         pos = nx.spring_layout(G, k=0.5, iterations=50)
    
#     if topology != '2d_modular':
#         nx.draw(G, pos, node_color='lightblue', node_size=300, with_labels=True, font_size=8, font_weight='bold', arrows=True)
    
#     plt.title(f"{topology.replace('_', ' ').title()} Topology ({G.number_of_edges()} edges)")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close()

def run_solver(args):
    G, tri, trt = args
    linNet = LinearNetwork(G)
    solver = LinearNetworkSolver(linNet)
    num_nodes = G.number_of_nodes()
    
    if num_nodes < 5:
        print(f"Warning: Graph has only {num_nodes} nodes. Skipping this trial.")
        return float('inf')
    
    source_nodes = [0, 1]
    target_nodes = [num_nodes-2, num_nodes-1]
    ground_nodes = [2]
    
    K, costs = solver.perform_trial(
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        ground_nodes=ground_nodes,
        in_node=tri,
        out_node=trt,
        lr=0.05,
        steps=250000,
        debug=True,
        every_nth=5000
    )
    return costs[-1][1]

def process_topology(args):
    topology, target_edges, tri, trt = args
    print(f"Testing topology: {topology} with target {target_edges} edges")
    trial_costs = []
    trial_edges = []
    for _ in range(5):  # 5 trials per topology and size
        G = create_topology_network(target_edges, topology)
        actual_edges = G.number_of_edges()
        final_cost = run_solver((G, tri, trt))
        trial_costs.append(final_cost)
        trial_edges.append(actual_edges)
    return topology, target_edges, trial_costs, trial_edges

if __name__ == "__main__":
    topologies = ['2d_modular', 'small_world', 'square_lattice', 'layer']
    
    # Generate and save sample figures
    sample_target_edges = 200
    os.makedirs("topology_images", exist_ok=True)
    for topology in topologies:
        filename = f"topology_images/{topology}.png"
        visualize_and_save_topology(topology, sample_target_edges, filename)
        print(f"Saved {filename}")
    
    # Generate random regression data
    tri, trt, tei, tet = generate_regression_data_for_experiment()

    # Define target number of edges
    target_edges_list = np.logspace(np.log10(50), np.log10(6000), num=20, dtype=int)

    results = {topology: {'costs': [], 'edges': []} for topology in topologies}

    # Prepare arguments for multiprocessing
    args_list = [(topology, target_edges, tri, trt) 
                 for target_edges in target_edges_list 
                 for topology in topologies]

    # Use multiprocessing to run trials
    with Pool(processes=int(0.8*cpu_count())) as pool:
        process_results = pool.map(process_topology, args_list)

    # Organize results
    for topology, target_edges, trial_costs, trial_edges in process_results:
        idx = np.where(target_edges_list == target_edges)[0][0]
        results[topology]['costs'].insert(idx, trial_costs)
        results[topology]['edges'].insert(idx, trial_edges)


    # Plotting
    plt.figure(figsize=(12, 8))
    # for topology in topologies:
    #     mean_costs = stats.gmean(results[topology]['costs'], axis=1)
    #     std_costs = np.std(np.log(results[topology]['costs']), axis=1)
    #     mean_edges = np.mean(results[topology]['edges'], axis=1)
    #     inv_edges = 1 / mean_edges
    #     plt.errorbar(inv_edges, mean_costs, yerr=std_costs, capsize=5, label=topology)
    for topology in topologies:
        mean_costs = stats.gmean(results[topology]['costs'], axis=1)
        log_mean_costs = np.log(mean_costs)
        log_std_costs = np.std(np.log(results[topology]['costs']), axis=1)
        mean_edges = stats.gmean(results[topology]['edges'], axis=1)  # Using geometric mean for consistency
        inv_edges = 1 / mean_edges
        
        lower_bound = np.exp(log_mean_costs - log_std_costs)
        upper_bound = np.exp(log_mean_costs + log_std_costs)
        
        plt.fill_between(inv_edges, lower_bound, upper_bound, alpha=0.3)
        plt.plot(inv_edges, mean_costs, label=topology)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1/N (Inverse Number of Edges)')
    plt.ylabel('Final Cost (Geometric Mean)')
    plt.title('Final Cost vs 1/N for Different Topologies')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("topology_analysis_plot.png")
    plt.close()

    # Save plotting data
    for topology in topologies:
        np.save(f"{topology}_costs.npy", np.array(results[topology]['costs']))
        np.save(f"{topology}_edges.npy", np.array(results[topology]['edges']))

    # Print LaTeX table
    print("Topology & " + " & ".join(f"{n}" for n in target_edges_list) + " \\\\")
    for topology in topologies:
        mean_costs = stats.gmean(results[topology]['costs'], axis=1)
        std_costs = np.std(np.log(results[topology]['costs']), axis=1)
        mean_edges = np.mean(results[topology]['edges'], axis=1)
        print(f"{topology} & {' & '.join(f'{cost:.4f} ({std:.4f}) [{edge:.0f}]' for cost, std, edge in zip(mean_costs, std_costs, mean_edges))} \\\\")

# Commented out code to read np files for later use
'''
def load_plotting_data(topologies):
    loaded_results = {}
    for topology in topologies:
        loaded_results[topology] = {
            'costs': np.load(f"{topology}_costs.npy"),
            'edges': np.load(f"{topology}_edges.npy")
        }
    return loaded_results

# Example usage:
# topologies = ['2d_modular', 'small_world', 'square_lattice', 'layer']
# loaded_results = load_plotting_data(topologies)

# Plotting with loaded data
# plt.figure(figsize=(12, 8))
# for topology in topologies:
#     mean_costs = stats.gmean(loaded_results[topology]['costs'], axis=1)
#     std_costs = np.std(np.log(loaded_results[topology]['costs']), axis=1)
#     mean_edges = np.mean(loaded_results[topology]['edges'], axis=1)
#     inv_edges = 1 / mean_edges
#     plt.errorbar(inv_edges, mean_costs, yerr=std_costs, capsize=5, label=topology)
# 
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('1/N (Inverse Number of Edges)')
# plt.ylabel('Final Cost (Geometric Mean)')
# plt.title('Final Cost vs 1/N for Different Topologies')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("loaded_topology_analysis_plot.png")
# plt.close()
'''


# Topology & 49 & 64 & 82 & 106 & 136 & 176 & 226 & 291 & 375 & 482 & 621 & 799 & 1028 & 1323 & 1702 & 2189 & 2817 & 3624 & 4663 & 5999 \\
# 2d_modular & 0.0000 (1.8595) [40] & 0.0000 (8.2647) [40] & 0.0000 (4.5358) [40] & 0.0000 (3.9325) [40] & 0.0000 (9.7396) [40] & 0.0000 (2.1581) [168] & 0.0000 (1.7511) [168] & 0.0000 (1.8642) [512] & 0.0000 (1.7464) [512] & 0.0000 (1.0770) [1240] & 0.0001 (2.3717) [2568] & 0.0004 (1.9550) [4760] & 0.0014 (0.9955) [8128] & 0.0018 (1.0009) [13032] & 0.0011 (0.9045) [19880] & 0.0032 (0.0129) [29128] & 0.0033 (0.0022) [56888] & 0.0033 (0.0023) [100920] & 0.0033 (0.0016) [166600] & 0.0033 (0.0009) [260072] \\
# small_world & 0.0000 (24.3229) [18] & 0.0000 (22.3979) [22] & 0.0000 (26.1310) [24] & 0.0000 (14.8237) [28] & 0.0000 (20.6752) [32] & 0.0000 (2.4600) [36] & 0.0000 (24.3911) [42] & 0.0000 (23.3699) [48] & 0.0000 (8.6920) [54] & 0.0000 (20.7419) [62] & 0.0000 (4.7658) [70] & 0.0000 (25.9366) [78] & 0.0000 (2.2996) [90] & 0.0000 (21.2791) [102] & 0.0000 (20.9395) [116] & 0.0000 (21.0745) [132] & 0.0000 (6.9773) [150] & 0.0000 (23.8142) [170] & 0.0001 (2.1184) [192] & 0.0000 (21.6861) [218] \\
# square_lattice & 0.0064 (0.0396) [24] & 0.0054 (0.2382) [40] & 0.0051 (0.3033) [60] & 0.0049 (0.2753) [84] & 0.0048 (0.2305) [112] & 0.0043 (0.2261) [144] & 0.0048 (0.1947) [180] & 0.0039 (0.0988) [264] & 0.0039 (0.0882) [312] & 0.0042 (0.0401) [420] & 0.0040 (0.0761) [544] & 0.0039 (0.0709) [684] & 0.0037 (0.0468) [924] & 0.0039 (0.0776) [1200] & 0.0038 (0.0550) [1624] & 0.0039 (0.0564) [2112] & 0.0040 (0.0724) [2664] & 0.0039 (0.0580) [3444] & 0.0039 (0.0877) [4512] & 0.0037 (0.0474) [5724] \\
# layer & 0.0000 (2.7931) [48] & 0.0000 (2.6502) [48] & 0.0003 (1.5094) [75] & 0.0000 (0.8563) [75] & 0.0004 (1.5505) [108] & 0.0031 (0.0458) [147] & 0.0019 (0.9577) [192] & 0.0031 (0.0441) [243] & 0.0033 (0.0159) [363] & 0.0032 (0.0026) [432] & 0.0033 (0.0029) [588] & 0.0033 (0.0042) [768] & 0.0033 (0.0034) [972] & 0.0033 (0.0030) [1323] & 0.0033 (0.0021) [1587] & 0.0033 (0.0030) [2187] & 0.0033 (0.0003) [2700] & 0.0033 (0.0005) [3468] & 0.0033 (0.0010) [4563] & 0.0033 (0.0019) [5808] \\