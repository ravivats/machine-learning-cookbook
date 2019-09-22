import numpy as np

# function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    eL = np.exp(L)
    return np.divide(eL, eL.sum())

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))    