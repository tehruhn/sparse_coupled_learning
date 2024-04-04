import numpy as np
np.random.seed(0)

def generate_regression_data():

    input_pairs = np.random.uniform(0, 1, (420, 2))

    VD1 = 0.15 * input_pairs[:, 0] + 0.20 * input_pairs[:, 1]
    VD2 = 0.25 * input_pairs[:, 0] + 0.10 * input_pairs[:, 1]

    targets = np.vstack((VD1, VD2)).T

    train_inputs = input_pairs[:400]
    test_inputs = input_pairs[400:]

    train_targets = targets[:400]
    test_targets = targets[400:]

    return (train_inputs, train_targets, test_inputs, test_targets)

def encode_regression_data_in_correct_format(split="train"):
    """
    Can be encoded as purely a node constraint
    Returns (node input data, node output data)
    """
    inputs, outputs = None, None
    train_inputs, train_targets, test_inputs, test_targets = generate_regression_data()
    if split == "train":
        inputs, outputs = train_inputs, train_targets
    else:
        inputs, outputs = test_inputs, test_targets
    return (train_inputs, train_targets) 
    

if __name__ == "__main__":
    tri, trt, tei, tet = generate_regression_data()