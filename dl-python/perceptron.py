import numpy as np

# Setting the random seed, can be tweaked
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# The function receives as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
#  It updates the weights W and bias b, according to the perceptron algorithm,
# and returns W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        addOrSubFactor = y[i] - y_hat 
        for j in range(2):
            W[j] += addOrSubFactor * X[i][j] * learn_rate
        b += addOrSubFactor * learn_rate
    return W, b
      
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# The learning rate and the num_epochs can be tweaked
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 1000):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))

    return boundary_lines