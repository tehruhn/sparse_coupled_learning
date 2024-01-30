from sparseflow import *

# print(create_square_grid_graph(3,3,True))

# Example usage
a, b = 100, 100  # Grid dimensions
periodic = False  # Periodic grid
trials = 1  # Number of trials
nTaskTypes = 2  # Number of task types
nTasksPerType = 1  # Number of tasks per type
eta = 1e-3  # Learning rate parameter
lr = 3.0  # Learning rate
Steps = 40001  # Number of learning steps
sources = 5  # Number of source nodes
targets = 3  # Number of target nodes
sourceedges = 1  # Number of source edges
targetedges = 1  # Number of target edges

costs, conductances = learning_process(a, b, periodic, trials, nTaskTypes, nTasksPerType, eta, lr, Steps, sources, targets, sourceedges, targetedges)

# Print final costs and conductances
print("Final costs:", costs)
print("Final conductances:", conductances)
