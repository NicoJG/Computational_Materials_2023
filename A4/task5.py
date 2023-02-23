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

spin = {"O2":1.0, "CO":0.0}
symmetry_number = {"O2":2, "CO":1}
magmom = {"O2":[1.7,1.7], "CO":[2.5,1.7]}

for molec in ["O2","CO"]:
    atoms = Atoms(molec, [(0, 0, 0), (1.2, 0, 0)])
    atoms.set_cell([12.0,12.0,12.0])
    atoms.set_initial_magnetic_moments(magmom[molec])
    atoms.center()
    atoms.pbc = True
    atoms.calc = GPAW(xc = 'PBE',
                    mode=PW(450),
                    kpts =(1, 1, 1),
                    spinpol=True,
                    symmetry={'point_group': False}, # Turn off point-group symmetry
                    txt = f"task5output/gpaw_output_{molec}.txt")
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

    parprint(f"Vibrational energies of {molec}: {vib_energies}")
    parprint(f"Potential Energy of {molec}: {potentialenergy:.5f} eV")
    parprint(f"Entropy of {molec}: {S:.5f} eV/K")
    parprint(f"Entropy of {molec}: {S*mol/J:.5f} J/mol K")
    parprint(f"Total runtime: {time.time()-start_time}")
