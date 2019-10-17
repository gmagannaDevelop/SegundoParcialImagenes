
import numpy as np

import skimage
import skimage.data
import skimage.morphology
import skimage.filters

def filtro_disco(image: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    
    """
    _circle = skimage.morphology.disk(radius)
    _filtered = skimage.filters.rank.mean(image, selem=_circle)
    return _filtered



# load example image
original = skimage.data.camera()

# create disk-like filter footprint with given radius
radius = 10
circle = skimage.morphology.disk(radius)

# apply median filter with given footprint = structuring element = selem
filtered = skimage.filters.median(original, selem = circle)

filtered2 = filtro_disco(original)

print(f'{list(map(lambda x: x.dtype, [original, filtered, filtered2]))}')


