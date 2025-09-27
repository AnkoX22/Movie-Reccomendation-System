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

    for i in range(len(X)):
        dot_product += X[i] * Y[i]
        x_squared += X[i] * X[i]
        y_squared += Y[i] * Y[i]

    denominator = np.sqrt(x_squared * y_squared)

    if denominator == 0:
        return 0

    return dot_product / denominator

def np_pearson_correlation(X, Y):
    X_cebtered = X - X.mean()
    Y_cebtered = Y - Y.mean()

    numerator = np.sum(X_cebtered * Y_cebtered)
    denominator = np.sqrt(np.sum(X_cebtered**2))*np.sqrt(np.sum(Y_cebtered**2))

    if denominator == 0:
        return 0
    return numerator / denominator

def manual_pearson_correlation(X, Y):

    X = np.array(X)
    Y = np.array(Y)

    if(len(X) != len(Y)):
        raise ValueError("Vectors must be the same length")

    if len(X) == 0:
        return 0

    X_total = 0
    Y_total = 0

    for i in range(len(X)):
        X_total += X[i]
        Y_total += Y[i]

    X_average = X_total / len(X)
    Y_average = Y_total / len(Y)

    X_centered = X - X_average
    Y_centered = Y - Y_average

    numerator = np.sum(X_centered * Y_centered)
    denominator = np.sqrt(np.sum(X_centered**2))*np.sqrt(np.sum(Y_centered**2))

    return numerator / denominator

