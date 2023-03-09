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
    parprint(f"Entropies for {molec}...")
    S_dict[molec] = []
    for T_i in tqdm(T, disable=(world.rank!=0)):
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
        S = thermo.get_entropy(temperature=300, pressure=100000)

        S_dict[molec].append(S)

        parprint(f"---------------------------")
        parprint(f"Molecule: {molec}")
        parprint(f"Temperature: {T_i:.2f} K")
        parprint(f"Vibrational energies: {vib_energies}")
        parprint(f"Potential Energy: {potentialenergy:.5f} eV")
        parprint(f"Entropy: {S:.5f} eV/K")
        parprint(f"Entropy: {S*mol/J:.5f} J/mol K")
        parprint(f"Total runtime: {time.time()-start_time}")

if world.rank == 0:
    data = list(zip(T, S_dict["O"], S_dict["CO"]))
    np.savetxt("task7output/task7_entropies.csv", data, header="# T[K], S_O[eV/K], S_CO[eV/K]", comments="", fmt= "%.10f, %.10f, %.10f")
    print("Runtime",time.time()-start_time)