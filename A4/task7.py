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

S_O2 = results["entropy_O2"]
S_CO = results["entropy_CO"]

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
    return np.exp(-E_a/(k_B*T))

def r_3(T, theta_O_, theta_CO_, E_a):
    return theta_O_*theta_CO_*k_3(T,E_a)

# %%
theta_O_dict = {}
theta_CO_dict = {}
r_3_dict = {}

for metal in ["Au","Pt","Rh"]:
    T = np.linspace(100,2000, 1901)

    E_ads_O = results[f"E_ads_{metal}_O"]
    E_ads_CO = results[f"E_ads_{metal}_CO"]
    E_a = results[f"E_a_{metal}"]

    theta_O_ = theta_O(T, E_ads_O, E_ads_CO)
    theta_CO_ = theta_CO(T, E_ads_O, E_ads_CO)
    r_3_ = r_3(T, theta_O_, theta_CO_, E_a)
    
    theta_O_dict[metal] = theta_O_
    theta_CO_dict[metal] = theta_CO_
    r_3_dict[metal] = r_3_

# %%
plt.figure()
for metal in ["Au","Pt","Rh"]:
    plt.plot(T, theta_O_dict[metal], label=metal)
plt.yscale("log")
plt.xlabel("Temperature (K)")
plt.ylabel("Fractional coverage of O")
plt.grid()
plt.legend()
plt.savefig("plots/task7_theta_O.pdf")

plt.figure()
for metal in ["Au","Pt","Rh"]:
    plt.plot(T, theta_CO_dict[metal], label=metal)
plt.yscale("log")
plt.xlabel("Temperature (K)")
plt.ylabel("Fractional coverage of CO")
plt.grid()
plt.legend()
plt.savefig("plots/task7_theta_CO.pdf")

plt.figure()
for metal in ["Au","Pt","Rh"]:
    plt.plot(T, r_3_dict[metal], label=metal)

plt.ylim(bottom=r_3_dict["Au"].min()*100, top=r_3_dict["Pt"].max()*10)
plt.yscale("log")
plt.xlabel("Temperature (K)")
plt.ylabel(r"Formation rate of $\mathrm{CO}_2$")
plt.grid()
plt.legend()
plt.savefig("plots/task7_r_3.pdf")


# %%
