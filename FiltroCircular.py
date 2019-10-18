
import copy
import seaborn as sns

from myfunctions import *

def grafica_diccionario(my_dict: dict, 
                         titulo: str, 
                        savefig: bool = True, 
                           name: str = 'radio_dif_potencia.png') -> None:
  """
    Hacer un scatterplot a partir de un diccionario.
  """
  x = list(my_dict.keys())
  y = list(my_dict.values())

  sns.set_style('darkgrid')
  plt.figure()
  plt.title(titulo)
  sns.scatterplot(x, y)
  plt.show()
  if savefig:
    plt.savefig(name)
##

def image_save_and_show(imagen: np.ndarray, titulo: str, archivo: str) -> None:
  """
  """
  plt.figure()
  plt.title(titulo)
  plt.imshow(imagen, cmap='gray')
  img.imsave(archivo, imagen, cmap='gray')
  plt.show()
##


def main():
  # Cargar imagen :
  I = img.imread('docs/FigP0401(test_pattern).tif')
  # Si no hacemos una copia de los valores a un nuevo
  # numpy.ndarray, el filtro circular genera un error.
  I = copy.deepcopy(I) 
  # Filtro pasabajos del inciso b.
  sigma = 16
  kind = 'low'
  IB = FiltraGaussiana(I, sigma=sigma, kind=kind)
  # Potencia de la imagen filtrada.
  pot_IB = ImPotencia(IB)

  # Diccionario para hacer la búsqueda en línea del radio
  # que minimice la diferencia de las potencias.
  search = {}

  # Ciclo iterando diversos radios [0,40)
  for r in range(40):
    _tmp     = filtro_disco(I, radius=r)
    _pot_tmp = ImPotencia(_tmp)
    delta    = np.abs(_pot_tmp - pot_IB)
    search.update({
      delta: r
    })
    print(f'radio: {r}, Δ(potencia) = {delta}')
  
  plot = {v:k for k,v in search.items()}
  grafica_diccionario(plot, 'radio del disco, abs(ΔPotencia)')

  R = search[min(search.keys())]
  imDisc = filtro_disco(I, radius=R)
  

  image_save_and_show(imDisc, f'Filtro de disco, radio = {R}', 'filtro_disco_ganador.png') 
  image_save_and_show(IB, f'Filtro gaussiano tipo={kind}, sigma={sigma}', 'filtro_pasabajos_gaussiano.png') 
  
##

if __name__ == "__main__":
    main()

