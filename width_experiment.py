import json
import numpy as np
from experiment_utils import create_network
from task_utils import generate_regression_data
from LinearNetwork import LinearNetwork
from LinearNetworkSolver import LinearNetworkSolver

np.random.seed(42)

# hidden_nodes_range_list = list(range(100, 10001, 100))
# hidden_nodes_range_list = list(range(2, 100, 2))
hidden_nodes_range_list = [60]

all_costs = {}

# Initialize data generation
tri, trt, tei, tet = generate_regression_data()

def save_costs(all_costs):
    # Helper function to save costs to a file
    with open('all_sizes.json', 'w') as f:
        json.dump(all_costs, f)

# # For zero hidden nodes
# print("hidden nodes", 0)
# source, target, num_hidden = 3, 2, 0
# G, S, H, T = create_network(source, 10, target, num_hidden)
# linNet = LinearNetwork(G)
# solver = LinearNetworkSolver(linNet)

# K, costs = solver.perform_trial(source_nodes=S[1:], 
#                                 target_nodes=T,
#                                 ground_nodes=[S[0]],
#                                 in_node=tri,
#                                 out_node=trt,
#                                 lr=1.e-3,
#                                 steps=10000
#                                 )
# all_costs[0] = costs

# Save after every completion of perform_trial for zero hidden nodes
# save_costs(all_costs)

# For others
for hidden_nodes in hidden_nodes_range_list:
    print("hidden nodes", hidden_nodes)
    source, target, num_hidden = 3, 2, 1
    G, S, H, T = create_network(source, hidden_nodes, target, num_hidden)
    # network_dict = {}
    # network_dict['NN'] = G.number_of_nodes()
    # network_dict['NE'] = G.number_of_edges()
    # network_dict['EI'] = [edge[0] for edge in G.edges()]
    # network_dict['EJ'] = [edge[1] for edge in G.edges()]
    # f = open("graph_dict_tarun.txt", "w")
    # f.write(str(network_dict))
    # f.close()
    # print("Wrote")
    linNet = LinearNetwork(G)
    solver = LinearNetworkSolver(linNet)
    
    K, costs = solver.perform_trial(source_nodes=S[1:], 
                                    target_nodes=T,
                                    ground_nodes=[S[0]],
                                    in_node=tri,
                                    out_node=trt,
                                    lr=1.e-2,
                                    steps=50000,
                                    debug=True,
                                    every_nth=1000
                                    )
    # all_costs[hidden_nodes] = costs

    # # Save after every completion of perform_trial for each hidden_nodes configuration
    # save_costs(all_costs)
