import json
import numpy as np
from experiment_utils import create_network, generate_random_regression_data
from task_utils import generate_regression_data
from LinearNetwork import LinearNetwork
from LinearNetworkSolver import LinearNetworkSolver

import matplotlib.pyplot as plt

hidden_nodes_range_list = [20]
all_costs = {}

# Initialize data generation
tri, trt, tei, tet = generate_regression_data()

def save_costs(all_costs):
    # Helper function to save costs to a file
    with open('all_sizes.json', 'w') as f:
        json.dump(all_costs, f)

for hidden_nodes in hidden_nodes_range_list:
    print("hidden nodes", hidden_nodes)
    source, target, num_hidden = 3, 2, 1
    G, S, H, T = create_network(source, hidden_nodes, target, num_hidden)

    linNet = LinearNetwork(G)
    solver = LinearNetworkSolver(linNet)
    
    K, costs = solver.perform_trial(source_nodes=S[0:-1], 
                                    target_nodes=T,
                                    ground_nodes=[S[-1]],
                                    in_node=tri,
                                    out_node=trt,
                                    lr=0.05,
                                    steps=250000,
                                    debug=True,
                                    every_nth=1000,
                                    init_strategy="random"
                                    )
    x, y = zip(*costs)
    y = [a/y[0] for a in y]
    plt.plot(x, y, color='blue')
    plt.title("Rel Cost vs Iter")
    plt.xlabel("Iter")
    plt.ylabel("Rel Cost")
    plt.yscale('log')
    plt.show()
    # all_costs[hidden_nodes] = costs

    # # Save after every completion of perform_trial for each hidden_nodes configuration
    # save_costs(all_costs)
