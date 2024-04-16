import numpy as np

def generate_random_regression_data(n_inputs, n_outputs, n_samples=420, train_split=0.95):
    """
    Generates random regression data
    """
    # generate random input pairs
    input_pairs = np.random.uniform(0, 1, (n_samples, n_inputs))
    
    # randomly generate coefficients for a simple linear model
    coefficients = np.random.uniform(0, 0.5, (n_inputs, n_outputs))
    
    # calculate targets based on the generated coefficients
    targets = np.dot(input_pairs, coefficients)
    
    # calculate split index for training and testing data
    split_index = int(n_samples * train_split)
    
    # split data into training and testing sets
    train_inputs = input_pairs[:split_index]
    test_inputs = input_pairs[split_index:]
    
    train_targets = targets[:split_index]
    test_targets = targets[split_index:]
    
    return (train_inputs, train_targets, test_inputs, test_targets)

def generate_regression_data_for_experiment():
    """
    Generates regression data for the specific experiment
    in the paper
    """
    input_pairs = np.random.uniform(0, 1, (420, 2))
    # Coeffs = array([[0.2,0.3],[0.1,0.5]]

    VD1 = 0.2 * input_pairs[:, 0] + 0.3 * input_pairs[:, 1]
    VD2 = 0.1 * input_pairs[:, 0] + 0.5 * input_pairs[:, 1]

    targets = np.vstack((VD1, VD2)).T

    train_inputs = input_pairs[:400]
    test_inputs = input_pairs[400:]

    train_targets = targets[:400]
    test_targets = targets[400:]

    return (train_inputs, train_targets, test_inputs, test_targets)