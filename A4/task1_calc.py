import time
start_time = time.time()

from tqdm.auto import tqdm
import numpy as np
from ase.build import bulk 
from gpaw import GPAW, PW
from ase.parallel import world,parprint
from ase.units import eV

a_range = np.linspace(3.5,4.5, 200)
energy_arrs = {"Au":[],"Pt":[],"Rh":[]}
elements = ["Au", "Pt", "Rh"]

for element in elements:
    parprint(f"Scan through lattice parameters for {element}...")
    for a0 in tqdm(a_range, disable=(world.rank!=0)):
        atoms = bulk(element, 'fcc', a=a0)
        atoms.calc = GPAW(xc = 'PBE',
                        mode=PW(450),
                        kpts =(12, 12, 12),
                        txt = None)
        energy = atoms.get_potential_energy() * eV
        energy_arrs[element].append(energy)
        parprint(f'element = {element}; a0 = {a0:.2f} Å; energy = {energy:.5f} eV')

if world.rank == 0:
    data = list(zip(a_range, energy_arrs["Au"], energy_arrs["Pt"], energy_arrs["Rh"]))
    np.savetxt("task1output/task1_energies.csv", data, header="# a0[Å], E_Au[eV], E_Pt[eV], E_Rh[eV]", comments="", fmt= "%.10f, %.10f, %.10f, %.10f")
    print("Runtime",time.time()-start_time)



