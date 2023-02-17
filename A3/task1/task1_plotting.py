# %%
import numpy as np
import matplotlib.pyplot as plt

# For latex interpretation of the figures
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 14.0
})

# %%
# The given H2O simulations
# read the given H2O trajectory
Time,Etot,Epot,Ekin,T = np.genfromtxt("../cluster24.log", skip_header=1, unpack=True)

plt.figure()
plt.plot(Time,T)
plt.xlabel("Time (ps)")
plt.ylabel("Temperature (K)")
plt.tight_layout()
plt.savefig("../plots/h2o_thermostat_temperature.pdf")

plt.figure()
plt.plot(Time,Etot)
plt.xlabel("Time (ps)")
plt.ylabel("Total energy (eV)")
plt.tight_layout()
plt.savefig("../plots/h2o_thermostat_energy.pdf")

# Our simulation (H2O+Na)
# read our H2O+Na trajectory
Time,Etot,Epot,Ekin,T = np.genfromtxt("mdOutput.log", skip_header=1, unpack=True)
plt.figure()
plt.plot(Time,T)
plt.xlabel("Time (ps)")
plt.ylabel("Temperature (K)")
plt.tight_layout()
plt.savefig("../plots/our_sim_temperature.pdf")

plt.figure()
plt.plot(Time,Etot)
plt.xlabel("Time (ps)")
plt.ylabel("Total energy (eV)")
plt.tight_layout()
plt.savefig("../plots/our_sim_energy.pdf")

# Given simulation (H2O+Na)
# read the given H2O+Na trajectory
Time,Etot,Epot,Ekin,T = np.genfromtxt("../NaCluster24.log", skip_header=1, unpack=True)
plt.figure()
plt.plot(Time,T)
plt.xlabel("Time (ps)")
plt.ylabel("Temperature (K)")
plt.tight_layout()
plt.savefig("../plots/given_sim_temperature.pdf")

plt.figure()
plt.plot(Time,Etot)
plt.xlabel("Time (ps)")
plt.ylabel("Total energy (eV)")
plt.tight_layout()
plt.savefig("../plots/given_sim_energy.pdf")
# %%
