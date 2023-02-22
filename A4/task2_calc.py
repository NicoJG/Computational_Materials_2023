# %%
# https://wiki.fysik.dtu.dk/gpaw/tutorialsexercises/energetics/rpa_ex/rpa.html
import numpy as np

# from task 1 in eV:
E_bulk = {
    "Au": -3.146,
    "Pt": -6.434,
    "Rh": -7.307
}

# from the task 2 description in eV:
E_at = {
    "Au": -0.158,
    "Pt": -0.704,
    "Rh": -1.128
}

for elmt in ["Au", "Pt", "Rh"]:
    #E_coh = E_at[elmt] - 0.5*E_bulk[elmt]
    E_coh = E_at[elmt] - E_bulk[elmt]
    print(f"Cohesive energy for {elmt}: {E_coh:.3f} eV")


# %%
