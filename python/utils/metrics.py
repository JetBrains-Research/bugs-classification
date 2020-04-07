import numpy as np


def cos_dist(x, y):
    product = np.dot(x, y)
    x_norm = np.sqrt(np.sum(x ** 2))
    y_norm = np.sqrt(np.sum(y ** 2))
    if x_norm == 0 or y_norm == 0:
        return 1
    return (1.0 - product / (x_norm * y_norm)) / 2.0
