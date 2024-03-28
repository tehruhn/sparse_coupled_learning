from experiment_utils import create_network, generate_regression_data
from LinearNetwork import LinearNetwork
from LinearNetworkSolver import LinearNetworkSolver

import numpy as np

if __name__ == "__main__":

    source, hidden, target, num_hidden = 3, 5, 2, 0
    G, S, H, T = create_network(source, hidden, target, num_hidden)
    tri, trt, tei, tet = generate_regression_data(2, 2)

    # make the linear network
    linNet = LinearNetwork(G)
    solver = LinearNetworkSolver(linNet)
    
    # add source, target, ground nodes
    source_nodes = np.array(S[1:], dtype=int)
    target_nodes = np.array(T, dtype=int)
    ground_nodes = np.array([S[0]], dtype=int)

    print(source_nodes, target_nodes, ground_nodes)

    # pass data to the network and train it
    K = solver.perform_trial(source_nodes=source_nodes, 
                            target_nodes=target_nodes,
                            ground_nodes=ground_nodes,
                            in_node=tri,
                            out_node=trt,
                            lr=1.e-3,
                            steps=10000
                            )