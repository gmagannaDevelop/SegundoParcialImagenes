#!/usr/bin/env python
# coding: utf-8

# # División de Ciencias e Ingenierías de la Universidad de Guanajuato
# ## Fundamentos de procesamiento digital de imágenes
# ## Segundo Examen Parcial
# ### Profesor : Dr. Arturo González Vega
# ### Alumno : Gustavo Magaña López

# ## Módulos necesarios :

# In[1]:


import numpy as np
import scipy.fftpack as F
import scipy.io as io

import cv2
import matplotlib.image as img

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


# ## Definición de funciones :

# In[20]:


def img_fft(image: np.ndarray, shift: bool = True) -> np.ndarray:
    """
        Ejecutar una Transformada de Fourier adecuada 
        para el procesamiento digital de imágenes.
        
        Parámetros :
                image : Imagen, representada como un arreglo de numpy (numpy.ndarray)
                shift : Booleano que indica si debe ejecutarse la traslación de la imagen e
                        en el espacio de frecuencia.
    """
    _X = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    if shift:
        _X_shift = np.fft.fftshift(_X)
    _X_complex = _X_shift[:,:,0] + 1j*_X_shift[:,:,1]
    _X_abs = np.abs(_X_complex) + 1 # Evitar que el logaritmo reciba 0 como argumento.
    _X_bounded = 20 * np.log(_X_abs)
    _X_img = 255 * _X_bounded / np.max(_X_bounded)
    _X_img = _X_img.astype(np.uint8)
    
    return _X_img
##

def ImPotencia(image: np.ndarray) -> float:
    """
    """
    
    return np.sum(np.abs(image)**2) / np.prod(image.shape)
##

def FiltraGaussiana(image: np.ndarray, kind: str = 'low', sigma: float) -> np.ndarray:
    """
    
    
    """
    kind   = kind.lower()
    _kinds = ['low', 'high', 'lowpass', 'highpass']
    if kind not in _kinds:
        raise Exception(f'Error : Invalid filter kind \"{kind}\". Use one of {_kinds}')
    
    _X = np.exp(-1.0 *)
    


# In[14]:


x = img.imread('docs/Fig.tif')
#print(f'{type(x[0][0])}')
plt.imshow(x, cmap='gray')


# In[9]:


X  = img_fft(x)


# In[10]:


plt.imshow(X, cmap='gray')


# In[11]:


ImPotencia(X)


# In[16]:


exception('eRROR')


# In[19]:


FiltraGaussiana(np.ndarray([]), kind='lowPalss')


# In[23]:


-1 * x[1][1]


# In[ ]:




