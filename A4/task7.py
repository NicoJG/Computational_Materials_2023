# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann

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
    return np.exp(-S_gas/k_B + (E_gas - E_ads)/(k_B*T))


def theta_O(T, E_ads_O, E_ads_CO):
    K_1 = K(T, S_O2, E_O2, E_ads_O)
    K_2 = K(T, S_CO, E_CO, E_ads_CO)
    return np.sqrt(K_1)/(1-K_2-np.sqrt(K_1))

def theta_CO(T, E_ads_O, E_ads_CO):
    K_1 = K(T, S_O2, E_O2, E_ads_O)
    K_2 = K(T, S_CO, E_CO, E_ads_CO)
    return K_2/(1+K_2-np.sqrt(K_1))

def k_3(T, E_a):
    nu = 1e12
    return np.exp(-E_a/(k_B*T))

def r_3(T, theta_O_, theta_CO_, E_a):
    return theta_O_*theta_CO_*k_3(T,E_a)

# %%
for metal in ["Au","Pt","Rh"]:
    T = np.linspace(100,2000, 1901)

    E_ads_O = results[f"E_ads_{metal}_O"]
    E_ads_CO = results[f"E_ads_{metal}_CO"]
    E_a = results[f"E_a_{metal}"]

    theta_O_ = theta_O(T, E_ads_O, E_ads_CO)
    theta_CO_ = theta_CO(T, E_ads_O, E_ads_CO)
    r_3_ = r_3(T, theta_O_, theta_CO_, E_a)

    plt.figure()
    plt.plot(T, theta_O_)
    #plt.yscale("log")
    plt.title(f"theta_O for {metal}")
    plt.show()

    plt.figure()
    plt.plot(T, theta_CO_)
    #plt.yscale("log")
    plt.title(f"theta_CO for {metal}")
    plt.show()

    plt.figure()
    plt.plot(T, r_3_)
    #plt.yscale("log")
    plt.title(f"r_3_ for {metal}")
    plt.show()


# %%
