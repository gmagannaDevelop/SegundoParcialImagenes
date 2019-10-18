
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as img

import skimage
import skimage.data
import skimage.morphology
import skimage.filters

from myfunctions import *

def filtro_disco(image: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    
    """
    _circle = skimage.morphology.disk(radius)
    _filtered = skimage.filters.rank.mean(image, selem=_circle)
    return _filtered
##

def main():
  pass
##

if __name__ == "main":
    main()

