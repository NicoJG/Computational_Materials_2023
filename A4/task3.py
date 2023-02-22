# %%
# https://wiki.fysik.dtu.dk/gpaw/tutorialsexercises/structureoptimization/surface/surface.html
# https://leppertlab.wordpress.com/2015/02/06/using-ase-to-create-surface-slabs/


from ase.build import fcc111
from gpaw import GPAW, PW
from ase.units import eV
from ase.parallel import world,parprint
from tqdm.auto import tqdm

E_bulk = {
    "Au": -3.146,
    "Pt": -6.434,
    "Rh": -7.307
}

a0 = {
    "Au": 4.178,
    "Pt": 3.967,
    "Rh": 3.837
}

for elmt in tqdm(["Au", "Pt", "Rh"], disable=(world.rank!=0)):
    slab = fcc111(elmt, a=a0[elmt], size=(3, 3, 3), vacuum=6.0)

    slab.calc = GPAW(xc = 'PBE',
                    mode=PW(450),
                    kpts =(4, 4, 1),
                    txt = f"task3output/gpaw_output_{elmt}.txt")

    E_surface = slab.get_potential_energy() * eV
    area = slab.get_volume() / slab.cell[2,2]
    gamma = (E_surface - E_bulk[elmt]*len(slab))/(2*area)

    parprint(f"Area ({elmt}): {area:.5f} Å")
    parprint(f"Slab Energy ({elmt}): {E_surface:.5f} eV")
    parprint(f"Surface Energy ({elmt}): {gamma:.5f} eV")

