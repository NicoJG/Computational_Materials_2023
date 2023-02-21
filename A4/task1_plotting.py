# %%
import numpy as np
import matplotlib.pyplot as plt

# For latex interpretation of the figures (if latex is available)
from shutil import which
if which("latexmk") is not None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern"
    })
plt.rcParams.update({"font.size":14.0})

# %%
a0, E_Au, E_Pt, E_Rh = np.genfromtxt("task1output/task1_energies.csv", unpack=True, delimiter=",")

da0 = a0[1] - a0[0]

E = {"Au":E_Au,"Pt":E_Pt,"Rh":E_Rh}

idx_min_Au = np.argmin(E_Au)
E_min_Au = E_Au[idx_min_Au]
a0_min_Au = a0[idx_min_Au]
idx_min_Pt = np.argmin(E_Pt)
E_min_Pt = E_Pt[idx_min_Pt]
a0_min_Pt = a0[idx_min_Pt]
idx_min_Rh = np.argmin(E_Rh)
E_min_Rh = E_Rh[idx_min_Rh]
a0_min_Rh = a0[idx_min_Rh]

plt.figure()
for i,element in enumerate(["Au","Pt","Rh"]):
    idx_min = np.argmin(E[element])
    E_min = E[element][idx_min]
    a0_min = a0[idx_min]

    plt.plot(a0, E[element], color=f"C{i}", label=f"{element} ({a0_min:.3f} Å, {E_min:.3f} eV)")
    plt.plot(a0_min,E_min, "x", color=f"C{i}")

plt.xlabel("lattice parameter (Å)")
plt.ylabel("energy (eV)")
plt.legend(title=rf"$\Delta a_0 = {da0:.5f}$ Å")
plt.grid()
plt.tight_layout()
plt.savefig(f"plots/task1_energy_lattice_parameter.pdf")

# %%
