# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann

# For latex interpretation of the figures (if latex is available)
from shutil import which
if which("latexmk") is not None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern"
    })
plt.rcParams.update({"font.size":14.0})

import json
 
with open("results.json", "r") as file:
    results = json.load(file)

k_B = 8.617333e-5 # eV/K

#S_O2 = results["entropy_O2"]
#S_CO = results["entropy_CO"]

T, S_O2, S_CO = np.genfromtxt("task7output/task7_entropies.csv", delimiter=",", unpack=True)

E_O2 = results["E_mol_O2"]
E_CO = results["E_mol_CO"]

# formulae
def K(T, S_gas, E_gas, E_ads):
    return np.exp(-S_gas/k_B + E_ads/(k_B*T))


def theta_O(T, E_ads_O, E_ads_CO):
    K_1 = K(T, S_O2, E_O2, 2*E_ads_O)
    K_2 = K(T, S_CO, E_CO, E_ads_CO)
    return np.sqrt(K_1)/(np.sqrt(K_1)+K_2+1)

def theta_CO(T, E_ads_O, E_ads_CO):
    K_1 = K(T, S_O2, E_O2, 2*E_ads_O)
    K_2 = K(T, S_CO, E_CO, E_ads_CO)
    return K_2/(np.sqrt(K_1)+K_2+1)

def k_3(T, E_a):
    nu = 1e12
    return nu*np.exp(-E_a/(k_B*T))

def r_3(T, theta_O_, theta_CO_, E_a):
    return theta_O_*theta_CO_*k_3(T,E_a)

# %%
theta_O_dict = {}
theta_CO_dict = {}
r_3_atomic_units_dict = {}
r_3_dict = {}

for metal in ["Au","Pt","Rh"]:
    E_ads_O = results[f"E_ads_{metal}_O"]
    E_ads_CO = results[f"E_ads_{metal}_CO"]
    E_a = results[f"E_a_{metal}"]

    theta_O_ = theta_O(T, E_ads_O, E_ads_CO)
    theta_CO_ = theta_CO(T, E_ads_O, E_ads_CO)
    r_3_ = r_3(T, theta_O_, theta_CO_, E_a)
    
    sites_per_area = 9/results[f"surface_area_{metal}"] # Å^-2

    mol = 6.02214076e23 # atoms
    m = 1e10 # Å

    theta_O_dict[metal] = theta_O_
    theta_CO_dict[metal] = theta_CO_
    r_3_atomic_units_dict[metal] = r_3_ # s^-1
    r_3_dict[metal] = r_3_*sites_per_area / mol * 1e20 # mol m^-2 s^-1

# %%
# Find the maxima
for metal in ["Au","Pt","Rh"]:
    idx_max = np.argmax(r_3_dict[metal])
    results[f"r_3_max_{metal}[mol Ang^-2 s^-1]"] = r_3_dict[metal][idx_max]
    results[f"T(r_3_max_{metal})[K]"] = T[idx_max]

with open("results.json",'w') as json_file:
    json.dump(results, json_file, indent=4)

# %%
plt.figure(figsize=(8,6))
for i,metal in enumerate(["Au","Pt","Rh"]):
    plt.plot(T, theta_O_dict[metal], f"C{i}--", label=fr"$\theta_O\,$ ({metal})")
    plt.plot(T, theta_CO_dict[metal], f"C{i}-", label=fr"$\theta_{{CO}}$({metal})")
plt.yscale("log")
plt.xlabel("Temperature (K)")
plt.ylabel("Fractional coverage")
plt.ylim(1e-15,10)
plt.grid()
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/task7_theta.pdf")

plt.figure(figsize=(8,6))
for metal in ["Au","Pt","Rh"]:
    plt.plot(T, r_3_atomic_units_dict[metal], label=metal)
plt.ylim(bottom=r_3_atomic_units_dict["Au"].min()*100, top=r_3_atomic_units_dict["Pt"].max()*10)
plt.yscale("log")
plt.xlabel("Temperature (K)")
plt.ylabel(r"Formation rate of $\mathrm{CO}_2$ ($\mathrm{s}^{-1}$)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("plots/task7_r_3_atomic_units.pdf")

plt.figure(figsize=(8,6))
for i,metal in enumerate(["Au","Pt","Rh"]):
    r_max = results[f"r_3_max_{metal}[mol Ang^-2 s^-1]"]
    T_max = results[f"T(r_3_max_{metal})[K]"]
    plt.plot(T, r_3_dict[metal], fr"C{i}-", label=f"{metal} ({T_max:.0f} K, {r_max:.3e})")
    plt.plot(T_max, r_max, f"C{i}x")
plt.ylim(bottom=r_3_dict["Au"].min()*100, top=r_3_dict["Pt"].max()*10)
plt.yscale("log")
plt.xlabel("Temperature (K)")
plt.ylabel(r"Formation rate of $\mathrm{CO}_2$ (mol $\mathrm{m}^{-2}$ $\mathrm{s}^{-1}$)")
plt.grid()
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/task7_r_3.pdf")



# %%
