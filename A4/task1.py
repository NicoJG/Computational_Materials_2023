import time
start_time = time.time()

import numpy as np
from ase.build import bulk 
from gpaw import GPAW, PW
import matplotlib.pyplot as plt

a_range = np.linspace(3.4,7, 10)
a_energy = []

for a0 in a_range:
    au = bulk('Au', 'fcc', a=a0)
    au.calc = GPAW( xc = 'PBE',
                    mode=PW(450),
                    kpts =(12, 12, 12),
                    txt = None) # from excercise txt=f'task1output/au{a0:.2f}_calculation.txt'
    energy = au.get_potential_energy()
    a_energy.append(energy)
    print(f'A0 for {a0:.2f} with energy {energy}')

plt.figure()
plt.plot(a_range, a_energy)
plt.savefig("task1output/a1-20.pdf")

data = list(zip(a_range, a_energy))
np.savetxt("task1output/au_energies.csv", data, header="a0, energy", comments="", fmt= "%1.10f, %1.10f")
print("Runtime",time.time()-start_time)



