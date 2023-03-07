from ase import Atoms
from ase.build import molecule
from gpaw import GPAW, PW
from ase.units import eV
from ase.parallel import world,parprint

import json

with open("results.json", "r") as file:
    results = json.load(file)

for molecule_str in ["O2","CO"]:
    atoms = molecule(molecule_str, vacuum=6.0)
    atoms.pbc = True

    atoms.calc = GPAW(xc = 'PBE',
                    mode=PW(450),
                    kpts =(1, 1, 1),
                    spinpol=True,
                    txt = f"task4output/gpaw_output_{molecule_str}.txt")

    E = atoms.get_potential_energy()*eV

    results[f"E_mol_{molecule_str}"] = E
    parprint(f"Energy of {molecule_str}: {E:.5f} eV")


with open("results.json",'w') as json_file:
    json.dump(results, json_file, indent=4)