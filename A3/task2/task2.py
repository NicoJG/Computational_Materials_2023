# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ase.io import read, Trajectory

trajectory = Trajectory("../task1/someDynamics.traj")

# %%
#from ase.geometry.analysis import Analysis
#
#nbins = 100
#rdf_sum = np.zeros(nbins)
#
#for atoms in tqdm(trajectory):
#    ana = Analysis(atoms)
#    rdf,dist = ana.get_rdf(rmax=4.4, nbins=nbins,return_dists=True, elements="O")[0]
#    rdf_sum += rdf

# %%
# Calculate the RDF
nbins = 100
rmin = 0.
rmax = 10

bin_edges = np.linspace(rmin,rmax,nbins)
bin_width = np.diff(bin_edges)[0]
bin_centers = bin_edges[:-1]+bin_width/2

def calc_rdf_of_timestep(atoms):
    elements = atoms.get_atomic_numbers()
    idx_Na = np.where(elements==11)[0]
    idxs_O = np.where(elements==8)
    
    dists = atoms.get_distances(idx_Na, idxs_O, mic=False)
    
    return np.histogram(dists, bin_edges)[0]

    
# parallelize the iteration through the trajectory
import multiprocessing as mp
n_cpus = mp.cpu_count()
with mp.Pool(processes = n_cpus) as p:
    r = list(tqdm(p.imap(calc_rdf_of_timestep, trajectory), total=len(trajectory)))  
    
bin_counts = np.sum(r,axis=0)
    
# %%
# Plot the RDF

plt.figure()
plt.hist(bin_centers,bin_edges,weights=bin_counts)
plt.xlabel("radial distance (Å)")
plt.ylabel("bin counts")
plt.show()

# %%
