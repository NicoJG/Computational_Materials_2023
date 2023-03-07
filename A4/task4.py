from ase import Atoms
from ase.build import molecule
from gpaw import GPAW, PW
from ase.units import eV
from ase.parallel import world,parprint

magmom = {"O2":[1.7,1.7], "CO":[2.5,1.7]}
bond_lengths = {"O2":1.16, "CO":1.43}

for molecule_str in ["O2","CO"]:
    atoms = Atoms(molecule, [(0, 0, 0), (1.2, 0, 0)])
    atoms.set_cell([12.0,12.0,12.0])
    atoms.set_initial_magnetic_moments(magmom[molecule_str])
    atoms.center()
    atoms.pbc = True

    atoms.calc = GPAW(xc = 'PBE',
                    mode=PW(450),
                    kpts =(1, 1, 1),
                    spinpol=True,
                    txt = f"task4output/gpaw_output_{molecule_str}.txt")

    E = atoms.get_potential_energy()*eV

    parprint(f"Energy of {molecule_str}: {E:.5f} eV")