import numpy as np
np.random.seed(0)

def generate_regression_data():

    input_pairs = np.random.uniform(1, 5, (420, 2))

    VD1 = 0.15 * input_pairs[:, 0] + 0.20 * input_pairs[:, 1]
    VD2 = 0.25 * input_pairs[:, 0] + 0.10 * input_pairs[:, 1]

    targets = np.vstack((VD1, VD2)).T

    train_inputs = input_pairs[:400]
    test_inputs = input_pairs[400:]

    train_targets = targets[:400]
    test_targets = targets[400:]

    return (train_inputs, train_targets, test_inputs, test_targets)

def encode_regression_data_in_correct_format(NN, NE, EI, EJ):
    """
    Can be encoded as purely a node constraint
    """
    pass