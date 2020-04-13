#@title Third party functions for saving model and plots

#il faut coder une classe Buffer

import numpy as np

pool_A, pool_B = [], []


def update_pool_A(image, max_size=50):
    selected = list()

    if len(pool_A) < max_size:
        # stock the pool
        pool_A.append(image)
        selected.append(image)
    elif np.random.random() < 0.5:
        # use image, but don't add it to the pool
        selected.append(image)
    else:
        # replace an existing image and use replaced image
        ix = np.random.randint(0, len(pool_A))
        selected.append(pool_A[ix])
        pool_A[ix] = image

    return selected

def update_pool_B(image, max_size=50):
    selected = list()

    if len(pool_B) < max_size:
        # stock the pool
        pool_B.append(image)
        selected.append(image)
    elif np.random.random() < 0.5:
        # use image, but don't add it to the pool
        selected.append(image)
    else:
        # replace an existing image and use replaced image
        ix = np.random.randint(0, len(pool_B))
        selected.append(pool_B[ix])
        pool_B[ix] = image
    
    return selected