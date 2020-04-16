#il faut coder une classe Buffer

import random as rd


class Buffer () :

  def __init__(self, max_size):
    self.max_size = max_size
    self.pool = []

  def update (self, image):
    selected = list()

    if len(self.pool) < self.max_size:
        # stock the pool
        self.pool.append(image)
        selected.append(image)
    elif rd.random() < 0.5:
        # use image, but don't add it to the pool
        selected.append(image)
    else:
        # replace an existing image and use replaced image
        ix = rd.randint(0, len(self.pool)-1)
        selected.append(self.pool[ix])
        self.pool[ix] = image

    return selected



#on la prendra si on a besoin
class Buffer_plus_souple() :

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


