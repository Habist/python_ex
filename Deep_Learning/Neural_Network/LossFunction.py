import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist, init_network




def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size # one-hotm
    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size # not one-hot