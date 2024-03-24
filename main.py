from sparseLinear import *

a, b = 15, 15 # Grid dimensions
periodic = False  # Periodic grid
trials = 1  # Number of trials
nTaskTypes = 2  # Number of task types
nTasksPerType = 1  # Number of tasks per type
eta = 1e-3  # Learning rate parameter
lr = 3.0  # Learning rate
Steps = 40001  # Number of learning steps
sources = 5  # Number of source nodes
targets = 3  # Number of target nodes
sourceedges = 2  # Number of source edges
targetedges = 2  # Number of target edges

np.random.seed(42)

perform_trials(nTaskTypes, nTasksPerType, eta, lr, Steps, 
                   a, b, sources, targets, sourceedges, targetedges, trials)