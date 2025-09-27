import numpy as np
import pandas as pd
import data_loader

def np_cosine_similarity(X, Y):
    return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))

def manual_cosine_similarity(X, Y):
    if len(X) != len(Y):
        raise ValueError("Vectors must be the same length")

    dot_product = 0
    x_squared = 0
    y_squared = 0

    for i in range(X.shape[0]):
        dot_product += X[i] * Y[i]
        x_squared += X[i] * X[i]
        y_squared += Y[i] * Y[i]

    denominator = np.sqrt(x_squared * y_squared)

    if denominator == 0:
        return 0

    return dot_product / denominator