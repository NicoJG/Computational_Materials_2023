# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ase.io import Trajectory

# For latex interpretation of the figures
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 14.0
})

trajectory = Trajectory("../cluster24.traj")[-6000:]

# %%

nbins = 200
rmin = 0.
rmax = 4.4

bin_edges = np.linspace(rmin,rmax,nbins+1)
bin_width = bin_edges[1]-bin_edges[0]
bin_centers = bin_edges[:-1]+bin_width/2

def calc_rdf_of_timestep(atoms):
    bin_counts = np.zeros_like(bin_centers)
    
    elements = atoms.get_atomic_numbers()
    idxs_O = np.where(elements==8)[0]
    
    for i in idxs_O:
        idxs_others = np.concatenate([idxs_O[:i],idxs_O[i+1:]])
        dist = atoms.get_distances(i, idxs_others, mic=True)
        
        bin_counts += np.histogram(dist, bin_edges)[0]
    
    # calculate the radial distribution function
    rho = bin_counts.sum()/(rmax**3*np.pi*4/3)
    rho_local = bin_counts/(4*np.pi*bin_width*bin_centers**2)
    rdf = rho_local/rho
    
    return rdf

import multiprocessing as mp
n_cpus = mp.cpu_count()
with mp.Pool(processes = n_cpus) as p:
    results = list(tqdm(p.imap(calc_rdf_of_timestep, trajectory), total=len(trajectory)))

print(np.array(results).shape)
rdf = np.mean(results, axis=0)

# calculate the correlation number
cut = 3.2

rho = 24/trajectory[-1].get_cell().volume
mask = bin_centers<cut
bin_centers_ = bin_centers[mask]
rdf_ = rdf[mask]
corr_number_first = 4*np.pi*rho*np.trapz(bin_centers_**2*rdf_, bin_centers_)

bin_centers__ = bin_centers[~mask]
rdf__ = rdf[~mask]
corr_number_second = 4*np.pi*rho*np.trapz(bin_centers__**2*rdf__, bin_centers__)

# %%
plt.figure()
plt.hist(bin_centers[mask], bin_edges, weights=rdf[mask], histtype="step", hatch="////", edgecolor="C0", label=f"first shell\n(integral={corr_number_first:.2f})")
plt.hist(bin_centers[~mask], bin_edges, weights=rdf[~mask], color="C1", histtype="step", hatch=r"\\\\", edgecolor="C1", label=f"second shell\n(integral={corr_number_second:.2f})")
#plt.hist(bin_centers, bin_edges, weights=rdf, color="C0", histtype="step")
plt.xlabel("Radial distance (Å)")
plt.ylabel("Radial distribution function")
plt.xlim(2.2,4.4)
plt.legend()
plt.tight_layout()
plt.savefig("../plots/h2o_rdf.pdf")
# %%
