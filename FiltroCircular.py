
import copy

from myfunctions import *

def main():
  I = img.imread('docs/FigP0401(test_pattern).tif')
  I = copy.deepcopy(I)
  IB = FiltraGaussiana(I, sigma=16, kind='low')
  pot_IB = ImPotencia(IB)

  search = {}

  for r in range(40):
    _tmp = filtro_disco(I, radius=r)
    _pot_tmp = ImPotencia(_tmp)
    print(f'radio: {r}, Δ(potencia) = {np.abs(_pot_tmp - pot_IB)}')
##

if __name__ == "main":
    main()

