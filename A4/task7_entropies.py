import time
start_time = time.time()
from ase.build import molecule
from gpaw import GPAW,PW
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from ase.parallel import world,parprint
from ase.units import eV, mol, J
from ase import Atoms
import json
import numpy as np
from tqdm import tqdm

spin = {"O2":1.0, "CO":0.0}
symmetry_number = {"O2":2, "CO":1}
    
T = np.linspace(100,2000,1901)

S_dict = {}

for molec in ["O2","CO"]:
    parprint(f"==========================")
    parprint(f"Start calculations for {molec}...")
    atoms = molecule(molec, vacuum=6.0)
    atoms.pbc = True
    atoms.calc = GPAW(xc = 'PBE',
                    mode=PW(450),
                    kpts =(1, 1, 1),
                    spinpol=True,
                    symmetry={'point_group': False}, # Turn off point-group symmetry
                    txt = None)
    potentialenergy = atoms.get_potential_energy()*eV

    vib = Vibrations(atoms)
    vib.run()
    vib_energies = vib.get_energies()

    thermo = IdealGasThermo(vib_energies=vib_energies,
                            potentialenergy=potentialenergy,
                            atoms=atoms,
                            geometry='linear', 
                            spin=spin[molec],
                            symmetrynumber = symmetry_number[molec])
    
    parprint(f"Vibrational energies: {vib_energies}")
    parprint(f"Potential Energy: {potentialenergy:.5f} eV")
    parprint(f"Entropies for {molec}...")

    S_dict[molec] = []
    for T_i in tqdm(T, disable=(world.rank!=0)):
        S = thermo.get_entropy(temperature=T_i, pressure=100000, verbose=False)
        S_dict[molec].append(S)
        parprint(f"Entropy at {T_i:.1f} K: {S:.5f} eV/K \t = {S*mol/J:.5f} J/mol K")

if world.rank == 0:
    data = list(zip(T, S_dict["O2"], S_dict["CO"]))
    np.savetxt("task7output/task7_entropies.csv", data, header="# T[K], S_O2[eV/K], S_CO[eV/K]", comments="", fmt= "%.10f, %.10f, %.10f")
    print("Runtime",time.time()-start_time)