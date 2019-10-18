
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

def main():
  # Cargar imagen :
  I = img.imread('docs/FigP0401(test_pattern).tif')
  # Si no hacemos una copia de los valores a un nuevo
  # numpy.ndarray, el filtro circular genera un error.
  I = copy.deepcopy(I) 
  # Filtro pasabajos del inciso b.
  IB = FiltraGaussiana(I, sigma=16, kind='low')
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
  
  plt.figure()
  plt.title(f'Filtro de disco, radio = {R}')
  plt.imshow(imDisc, cmap='gray')
  img.imsave('filtro_disco_ganador.png', imDisc)
  plt.show()

##

if __name__ == "__main__":
    main()

