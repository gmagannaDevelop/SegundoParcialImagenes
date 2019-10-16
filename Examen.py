#!/usr/bin/env python
# coding: utf-8

# # División de Ciencias e Ingenierías de la Universidad de Guanajuato
# ## Fundamentos de procesamiento digital de imágenes
# ## Segundo Examen Parcial
# ### Profesor : Dr. Arturo González Vega
# ### Alumno : Gustavo Magaña López

# ## Módulos necesarios :

# In[1]:


from typing import Tuple

import numpy as np
import scipy.fftpack as F
import scipy.io as io

import cv2
import matplotlib.image as img

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


# ## Definición de funciones :

# In[73]:


def img_fft(image: np.ndarray, shift: bool = True) -> np.ndarray:
    """
        Ejecutar una Transformada de Fourier visualizable con matplotlib.pyplot.imshow() .
        
        Basado en un snippet encontrado en :
        https://medium.com/@y1017c121y/python-computer-vision-tutorials-image-fourier-transform-part-2-ec9803e63993
        
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

def paddedsize(*args, **kwargs) -> Tuple[int]:
    """
        Traducción a Python3.7 de la función definida en:
        Gonzalez Image Processing with MATLAB
        Capítulo 4, Sección 3 :  'Filtering in the frequency domain', página 117
        
        Docstring (adaptación del original): 
        PADDEDSIZE Computes padded sizes useful for FFT-based filtering.
        PADDEDSIZE calcula los tamaños 'amortiguados' adecuados para filtrado FFT. 
        
        El parámetro 'PWR2' puede también ser especificado con minúsculas.
        i.e. 'pwr2'
        
        PQ = PADDEDSIZE(AB)
            Donde AB es una lista/vector/arreglo de dos elementos, 
            calcula el vector PQ tal que :
                PQ = 2 * AB
        
        PQ = PADDEDSIZE(AB, 'PWR2')
            Calcula el vector PQ tal que :
                PQ[0] = PQ[1] = 2**ceil(log2(abs(2*m)))
                donde m = max(AB)
                
        PQ = PADDEDSIZE(AB, CD)
            Donde AB y CD son listas/vectores/arreglos de dos elementos, 
            calcula el vector PQ. 
            Los elementos de PQ son los enteros pares más pequeños, 
            mayores o iguales que AB + CD - 1
            
        PQ = PADDEDSIZE(AB, CD, 'PWR2')
            Parecido a PADDEDSIZE(AB, 'PWR2'), sólo que
            toma en cuenta todos los valores contenidos
            tanto en AB, como en CD.
                P[0] = P[1] = 2**ceil(log2(abs(2*m)))
                donde m = max([*AB, *CD])
                
        PQ es de tipo 'Tuple', conteniendo 2 enteros.
    """
    
    nargin = len(args)
    
    if nargin == 0:
        print('Error, especificar al menos un parámetro.')
        print(paddedsize.__doc__)
        return None
    elif nargin == 1:
        try:
            AB = np.array(args[0])
        except:
            raise Exception(f'AB es un objeto de la clase {type(args[0])}, no puede ser convertido a np.array')
        PQ = 2 * AB
    elif nargin == 2 and type(args[1]) is not str:
        try:
            AB = np.array(args[0])
            CD = np.array(args[1])
        except:
            _e = f'AB es un objeto de la clase {type(args[0])}, CD de la clase {type(args[1])}\n'
            _e += 'Alguno de los dos no pudo ser convertido a np.array'
            raise Exception(_e)
        PQ = AB + CD - 1
        PQ = 2 * np.ceil(PQ / 2)
    elif nargin == 2 and type(args[1]) is str:
        try:
            AB = np.array(args[0])
        except:
            raise Exception(f'AB es un objeto de la clase {type(args[0])}, no puede ser convertido a np.array')
        m  = AB.max()
        P  = 2**np.ceil(np.log2(np.abs(2 * m)))
        PQ = [P, P]
    elif nargin == 3 and type(args[2]) is str:
        try:
            AB = np.array(args[0])
            CD = np.array(args[1])
        except:
            _e = f'AB es un objeto de la clase {type(args[0])}, CD de la clase {type(args[1])}\n'
            _e += 'Alguno de los dos no pudo ser convertido a np.array'
            raise Exception(_e)
        m  = max([*AB, *CD])
        P  = 2**np.ceil(np.log2(np.abs(2 * m)))
        PQ = [P, P]
    else:
        print('Error, número o tipo de parámetros incorrectos.')
        print(paddedsize.__doc__)
        return None
    
    return tuple(PQ)
##

def ImPotencia(image: np.ndarray) -> float:
    """
    """
    return np.sum(np.abs(image)**2) / np.prod(image.shape)
##

def fourier_meshgrid(image: np.ndarray):
    """
    """
    
    # Creamos el rango para las variables frecuenciales.
    u, v = list(map(lambda x: np.arange(0, x), image.shape))
    idx, idy = list(map(lambda x, y: np.nonzero(x > y/2), [u, v], image.shape))
##
    
def FiltraGaussiana(image: np.ndarray, sigma: float, kind: str = 'low') -> np.ndarray:
    """
    
    
    """
    kind   = kind.lower()
    _kinds = ['low', 'high', 'lowpass', 'highpass']
    if kind not in _kinds:
        raise Exception(f'Error : Tipo desconocido de filtro \"{kind}\".\n Tipos disponibles : {_kinds}')
    
    #_X = np.exp(-1.0 *)
##


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


# In[25]:


help(np.fft.fftshift)


# In[26]:


freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
freqs


# In[27]:


np.fft.fftshift(freqs)


# In[28]:


freqs.in freqs[ freqs > 3 ]


# In[45]:


#dir(freqs)


# In[31]:


list(map(lambda x, y: x > y, [1, 2], [1, 1, 1]))


# In[32]:


x.shape


# In[35]:


np.array(
    np.array([1.4, 2])
)


# In[36]:


_ = np.array([1, 2, 4, 5])


# In[40]:


_.shape = tuple(np.array([2, 2]))


# In[41]:


A = B = C = 3


# In[44]:


[*[1, 2], *[3, 4]]


# In[47]:


print(img_fft.__doc__)


# In[54]:


def lol(*args):
    """
    doc
    """
    print(f'args are : {args}')


# In[57]:


lol()


# In[59]:


y = np.array([1, 2, 3])


# In[61]:


[1, *y]


# In[62]:


type(x.shape)


# In[63]:


type('asd') is str


# In[67]:


y.max()


# In[70]:


np.ceil(np.log2(np.abs(4.1)))


# In[ ]:




