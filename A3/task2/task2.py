# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# For latex interpretation of the figures
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern",
    "font.size": 14.0
})

from ase.io import read, Trajectory

# %%
# from ase.geometry.analysis import Analysis

# nbins = 100
# rdf_sum = np.zeros(nbins)

# trajectory = Trajectory("../NaCluster24.traj")[6000:]

# for atoms in tqdm(trajectory):
#     ana = Analysis(atoms)
#     rdf,dist = ana.get_rdf(rmax=4.4, nbins=nbins,return_dists=True, elements="O")[0]
#     rdf_sum += rdf
    
# # %%
# plt.plot(dist,rdf_sum/8000)
# plt.show()

# first_minimum_idx = argrelextrema(rdf_sum, np.less, order=4)[0][0]
# print("first solvation shell integral (ase rdf): ",np.sum(rdf_sum[:first_minimum_idx]/8000)*np.diff(dist)[0])


# %%
# Calculate the RDF
from ase.geometry import wrap_positions
nbins = 200
rmin = 0.
rmax = 10.

bin_edges = np.linspace(rmin,rmax,nbins)
bin_width = np.diff(bin_edges)[0]
bin_centers = bin_edges[:-1]+bin_width/2

def calc_rdf_of_timestep(atoms):
    elements = atoms.get_atomic_numbers()
    idx_Na = np.where(elements==11)[0]
    idxs_O = np.where(elements==8)
    
    pos = atoms.get_positions(wrap=True)
    pos_Na = pos[idx_Na][0]
    pos_O = pos[idxs_O]
    
    # implement periodic boundary conditions by copying the O positions 26 times around the main cell
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    pos_all_O = [wrap_positions(pos_O,cell,pbc,center=[0.5+ix,0.5+iy,0.5+iz]) 
                 for ix in range(-1,2) 
                 for iy in range(-1,2) 
                 for iz in range(-1,2)]
    
    pos_all_O = np.concatenate(pos_all_O, axis=0)
    
    dists = (pos_Na[0]-pos_all_O[:,0])**2
    dists += (pos_Na[1]-pos_all_O[:,1])**2
    dists += (pos_Na[2]-pos_all_O[:,2])**2
    dists = np.sqrt(dists)
    
    bin_counts = np.histogram(dists, bin_edges)[0]
    
    # calculate the radial distribution function
    rho = bin_counts.sum()/(rmax**3*np.pi*4/3)
    #rho = 24/cell.volume
    
    rho_local = bin_counts/(4*np.pi*bin_width*bin_centers**2)
    g = rho_local/rho
    
    return g, rho

# Calculate the first solvation shell
from scipy.signal import argrelextrema

def find_first_solvation_shell(g,bin_centers,rho):
    bin_width = bin_centers[1]-bin_centers[0]
    first_minimum_idx = argrelextrema(g, np.less, order=4)[0][0]
    first_minimum = bin_centers[first_minimum_idx]
    corr_number = np.sum(g[:first_minimum_idx])*bin_width
    
    
    corr_number = 4*np.pi*rho*np.trapz(bin_centers[:first_minimum_idx]**2*g[:first_minimum_idx], bin_centers[:first_minimum_idx])
    return first_minimum_idx, corr_number

import multiprocessing as mp
def calc_rdf(filepath, eq_idx=0):
    # eq_idx determines at which index to start (after equilibration)
    trajectory = Trajectory(filepath)[eq_idx:]
    
    # parallelize the iteration through the trajectory
    n_cpus = mp.cpu_count()
    with mp.Pool(processes = n_cpus) as p:
        results = list(tqdm(p.imap(calc_rdf_of_timestep, trajectory), total=len(trajectory)))  
        
    g_arr = []
    rho_arr = []
    for res in results:
        g_arr.append(res[0])
        rho_arr.append(res[1])
        
    g = np.mean(g_arr, axis=0)
    rho = np.mean(rho_arr)
    
    print(rho)
    
    first_minimum_idx, corr_number = find_first_solvation_shell(g, bin_centers, rho)
    
    return g, first_minimum_idx, corr_number

# %%
g_our, first_minimum_idx_our, corr_number_our = calc_rdf("../task1/someDynamics.traj")
g_given, first_minimum_idx_given, corr_number_given = calc_rdf("../NaCluster24.traj", 6000)

# %%
# Plot the RDF
plt.figure()

plt.hist(bin_centers[:first_minimum_idx_our],bin_edges[:first_minimum_idx_our+1],weights=g_our[:first_minimum_idx_our], histtype="step", edgecolor="C0", alpha=0.3, hatch="////", label=f"first solvation shell\n(integral = {corr_number_our:.2f})")
plt.hist(bin_centers,bin_edges,weights=g_our, histtype="step", color="C0")

plt.hist(bin_centers[:first_minimum_idx_given],bin_edges[:first_minimum_idx_given+1],weights=g_given[:first_minimum_idx_given], histtype="step", edgecolor="C1", alpha=0.3, hatch=r"\\\\", label=f"first solvation shell\n(integral = {corr_number_given:.2f})")
plt.hist(bin_centers,bin_edges,weights=g_given, histtype="step", color="C1")

plt.xlim(2,6)
plt.ylim(bottom=0)
plt.xlabel("radial distance (Å)")
plt.ylabel("radial distribution function")
plt.legend()
plt.tight_layout()
plt.show()

# %%
