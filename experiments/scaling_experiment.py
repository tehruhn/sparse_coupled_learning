
import json
import os
import time
import psutil
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.graph_utils import create_random_network
from utils.data_utils import generate_regression_data_for_experiment

from CLSolver.LinearNetwork import LinearNetwork
from CLSolver.LinearNetworkSolver import LinearNetworkSolver

import matplotlib.pyplot as plt

def measure_performance(G, tri, trt):
    linNet = LinearNetwork(G)
    solver = LinearNetworkSolver(linNet)
    
    # Measure time and memory usage for one step of the solver
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    K, costs = solver.perform_trial(source_nodes=[0, 1], 
                                    target_nodes=[num_nodes-2, 
                                                  num_nodes-1],
                                    ground_nodes=[2],
                                    in_node=tri, 
                                    out_node=trt, 
                                    lr=0.05, 
                                    steps=10, 
                                    debug=True)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    execution_time = end_time - start_time
    memory_usage = end_memory - start_memory
    
    return execution_time, memory_usage

if __name__ == "__main__":
    
    # generate random regression data
    tri, trt, tei, tet = generate_regression_data_for_experiment()

    # define the range of graph sizes to test
    num_nodes_range = [100, 250, 500, 1000, 2000, 3000, 4000, 5000]
    num_edges_range = [elem*5 for elem in num_nodes_range]


    execution_times = []
    memory_usages = []

    for num_nodes, num_edges in zip(num_nodes_range, num_edges_range):
        print(f"Testing graph size: {num_nodes} nodes, {num_edges} edges")
        G = create_random_network(num_nodes, num_edges)
        execution_time, memory_usage = measure_performance(G, tri, trt)
        execution_times.append(execution_time)
        memory_usages.append(memory_usage)

    # Plot execution time vs graph size
    plt.figure(figsize=(8, 6))
    plt.plot(num_nodes_range, execution_times, marker='o')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Graph Size')
    plt.grid(True)
    plt.show()

    # Plot memory usage vs graph size
    plt.figure(figsize=(8, 6))
    plt.plot(num_nodes_range, memory_usages, marker='o')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Memory Usage (bytes)')
    plt.title('Memory Usage vs Graph Size')
    plt.grid(True)
    plt.show()