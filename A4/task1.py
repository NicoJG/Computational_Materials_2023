import numpy as np
from ase.build import bulk 
from gpaw import GPAW, PW

a_range = np.linspace(1,20,20)

for a0 in a_range:
    au = bulk('Au', 'fcc', a=a0)
    au.calc = GPAW( xc = 'PBE',
                    mode=PW(450),
                    kpts =(12, 12, 12),
                    txt=f'task1output/au{a0:2f}_calculation.txt')

