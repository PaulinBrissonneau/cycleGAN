#il faut coder une classe Buffer

import numpy as np
import random as rd


class Buffer_version1 () :

  def __init__(self, max_size):
    self.max_size = max_size
    self.pool = []

  def update (self, image):
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



class Buffer_version2 () :

  def __init__(self, max_size):
    self.max_size = max_size
    self.pool = []

  def get_image (self, new_image):
    if len(self.pool)== 0:
      return new_image
    if rd.random() < 0.5 :
      return new_image
    else :
      return rd.choice(self.pool)

  def update (self, image):
    if len(self.pool) < self.max_size :
      self.pool.append(image)
    else:
      pop = rd.randint(0, len(self.pool)-1)
      self.pool.pop(pop)
      self.pool.append(image)

