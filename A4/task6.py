# %%
from ase.build import fcc111, add_adsorbate, molecule
from ase.units import eV,J,m
from ase.parallel import world,parprint
from tqdm.auto import tqdm
from ase.visualize import view
from ase.io import write
from gpaw import GPAW, PW
from ase.optimize import GPMin
from ase.constraints import FixAtoms

from pathlib import Path
Path("task5output/structures").mkdir(exist_ok=True)

a0 = {
    "Au": 4.178,
    "Pt": 3.967,
    "Rh": 3.837
}

combinations = [("Au","O"),("Pt","O"),("Rh","O"),
                ("Au","CO"),("Pt","CO"),("Rh","CO")]

for metal,adsorbate in tqdm(combinations):
    atoms = fcc111(metal, a=a0[metal], size=(3, 3, 3), vacuum=6.0)
    atoms.pbc = True
    
    fixed_idxs = list(range(len(atoms)))
    
    if adsorbate == "O":
        add_adsorbate(atoms, "O", height=2.0, position="fcc")
    elif adsorbate == "CO":
        add_adsorbate(atoms, molecule("CO"), height=2.0, position="top")
    
    # Constrain all atoms except the adsorbate:
    atoms.constraints = [FixAtoms(indices=fixed_idxs)]
    
    write(f"task6output/structures/{metal}_{adsorbate}_before.xyz", atoms)
    
    atoms.calc = GPAW(xc = 'PBE',
                    mode=PW(450),
                    kpts =(12,12,12),
                    txt = f"task6output/gpaw_output_{metal}_{adsorbate}.txt")
    
    dyn = GPMin(atoms, trajectory=f'task6output/structures/relax_{metal}_{adsorbate}.traj', logfile=f'task6output/structures/relax_{metal}_{adsorbate}.log')
    dyn.run(fmax=0.1, steps=1000)
    
    write(f"task6output/structures/{metal}_{adsorbate}_after.xyz", atoms)
    
    
    




# %%
