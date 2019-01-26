import numpy as np

input_set = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
output_labels = np.array([[0, 1, 1, 0]]).T
np.random.seed(1)
edge_weights = 2 * np.random.random((3, 1)) - 1

for iteration in range(10000):
    output = 1 / (1 + np.exp(-(np.dot(input_set, edge_weights))))
    edge_weights += np.dot(input_set.T, (output_labels - output) * output * (1 - output))

test_set = np.array([1, 0, 0])
print(1 / (1 + np.exp(-(np.dot(test_set , edge_weights)))))
