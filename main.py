from flow import *
import time

# Example usage with random graph
N = 50 # Number of nodes
edge_probability = 0.05  # Probability of additional edges
edges = create_connected_random_graph(N, edge_probability)
conductances = [random.uniform(0.5, 1.5) for _ in edges]

# Construct the Laplacian matrix
L_flow = construct_flow_laplacian(N, edges, conductances)

# Define source and target nodes with their pressures
source_nodes = [random.randint(0, N-1) for _ in range(5)]
source_pressures = [random.uniform(-5, 5) for _ in source_nodes]
target_nodes = [random.randint(0, N-1) for _ in range(5)]
target_pressures = [random.uniform(-5, 5) for _ in target_nodes]

# Solve for free and clamped state pressures
start = time.time()
free_state_pressures = solve_for_free_state_pressures(L_flow, N, source_nodes, source_pressures)
clamped_state_pressures = solve_for_clamped_state_pressures(L_flow, N, source_nodes, source_pressures, target_nodes, target_pressures)
end = time.time()
print("Total time taken : ", end-start)

# Plot the network (example with free state pressures)
plot_flow_network(N, edges, conductances, free_state_pressures, source_nodes)