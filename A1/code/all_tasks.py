# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import os
if os.path.basename(os.getcwd()) == "code":
    os.chdir("..")
    print("Changed working directory!")
    
hartree_energy = 27.211386245988 # eV

# %%
###################################
# Task 2
###################################
# Task 2 procedure function
def solve_for_V_sH(r, u):
    n_r = len(r)
    dr = np.diff(r)[0]
    r_max = np.max(r)
    
    f = np.zeros_like(r)
    mask = r!=0
    f[mask] = -u[mask]**2/r[mask]
    f[0] = 0
    f[-1] = 0
    
    A = 1/dr**2*(np.diag(-2*np.ones(n_r), 0) \
        + np.diag(1*np.ones(n_r-1), 1) \
        + np.diag(1*np.ones(n_r-1), -1))
    A[0,0] = 1
    A[0,1] = 0
    A[-1,-1] = 1
    A[-1,-2] = 0
    
    U_0 = np.linalg.solve(A,f)
    V_sH = np.zeros_like(r)
    V_sH[mask] = (U_0[mask]+r[mask]/r_max)/r[mask]
    return V_sH

# %%
# Task 2 calculation
n_r = 1000
r_min = 0.
r_max = 10
r = np.linspace(r_min, r_max, n_r)
dr = np.diff(r)[0]

# hydrogen ground state
u = 2*r*np.exp(-r)
V_H = solve_for_V_sH(r,u)

V_H_exact = 1/r - (1+1/r)*np.exp(-2*r)

# %%
# Task 2 plotting
plt.figure(figsize=(5,4))
plt.plot(r,V_H, label="numerical solution")
plt.plot(r,V_H_exact, "--", label="exact")
plt.xlabel(r"$r \: / \:$atomic units")
plt.ylabel(r"$V_H \: / \:$atomic units")
plt.ylim(0,1.1)
plt.grid()
plt.legend()
plt.show()
plt.savefig("plots/task2.pdf")
#TODO: plot the difference to the exact potential

# %%
#############################
# Task 3
#############################
# Task 3 procedure functions
def solve_kohn_sham_eq(r, total_potential):
    n_r = len(r)
    dr = np.diff(r)[0]
    
    A = 1/dr**2*(np.diag(-2*np.ones(n_r), 0) \
        + np.diag(1*np.ones(n_r-1), 1) \
        + np.diag(1*np.ones(n_r-1), -1))

    M = -0.5*A + np.diag(total_potential,0) 
    M[0,0] = 1
    M[0,1] = 0
    M[-1,-1] = 1
    M[-1,-2] = 0
    
    E, u = np.linalg.eigh(M)
    E, u = E[1], u[:,1]
    
    # normalize u
    u /= -np.sqrt(np.trapz(u**2, r))
    
    return E, u

def calc_psi_from_u(r,u):
    mask = r!=0
    psi = np.zeros_like(u)
    psi[mask] = u[mask]/(np.sqrt(4*np.pi)*r[mask])
    # normalize psi
    psi[mask] /= -np.sqrt(np.trapz(4*np.pi*r[mask]**2*psi[mask]**2,r[mask]))
    # make it positive
    if np.trapz(psi,r)<0:
        psi *= -1
    return psi

# %%
# Task 3 calculations
n_r = 2000
r_min = 0.
r_max = 10.
r = np.linspace(r_min, r_max, n_r)
dr = np.diff(r)

potential = -1/r

E, u = solve_kohn_sham_eq(r, potential)
psi = calc_psi_from_u(r,u)

psi_exact = 1/np.sqrt(np.pi)*np.exp(-r)

# %%
# Task 3 plotting
plt.figure(figsize=(5,4))
plt.plot(r,psi_exact, label="psi exact")
plt.plot(r,psi, label=f"psi numerical")
plt.xlabel(r"$r \: / \:$atomic units")
plt.ylabel(r"$\Psi \: / \:$atomic units")
plt.ylim(-0.1,0.7)
plt.grid()
plt.legend()
plt.show()
plt.savefig("plots/task3.pdf")

# %%
#########################
# Task 4
#########################
# Task 4 procedure functions
def calc_E0_from_eigenvalue(E,u,r,V_H):
    return 2*E - 2*np.trapz(u**2*0.5*V_H, r)

def solve_helium_kohn_sham_eq(r, u0, eps, n_max_iter=20):
    n_r = len(r)
    r_max = r[-1]
    dr = r[1]-r[0]
    u = u0
    n_max_iter = 100
    E_arr = []
    u_arr = []
    for i in range(n_max_iter):
        V_sH = solve_for_V_sH(r, u)
        V_H = V_sH
        potential = -2/r + V_H
        E, u = solve_kohn_sham_eq(r, potential)
        E_arr.append(calc_E0_from_eigenvalue(E,u,r,V_H))
        u_arr.append(u)
        if len(E_arr)>1:
            if abs(E_arr[-1]-E_arr[-2]) < eps:
                return E_arr, u_arr, i

# %%
# Task 4 calculations
eps_kohn_sham = 1e-5 * hartree_energy # eV
eps_grid_spacing = 1e-4 * hartree_energy # eV
n_max_iter1 = 20
n_max_iter2 = 20

# initial grid
dr = 0.1
r_min = 0.
r_max = 2.
r = np.arange(r_min, r_max+dr, dr) # so that r_max is included

# initially choose the hydrogen ground state
u0 = 2*r*np.exp(-r)

E_arr, u_arr, i_max = solve_helium_kohn_sham_eq(r,u0, eps_kohn_sham)
r_max_arr = [r_max]
E_arr_arr = [E_arr]
i_arr = [i_max]

pbar = tqdm(range(n_max_iter1))
for i in pbar:
    r_max *= 1.5
    r = np.arange(r_min, r_max+dr, dr)
    n_r = len(r)
    pbar.set_description(f"r_max = {r_max:.1f}; n_r = {n_r}")
    u0 = 2*r*np.exp(-r)
    E_arr, u_arr, i_max = solve_helium_kohn_sham_eq(r, u0, eps_kohn_sham)
    r_max_arr.append(r_max)
    E_arr_arr.append(E_arr)
    i_arr.append(i_max)
    if abs(E_arr_arr[-1][-1]-E_arr_arr[-2][-1]) < eps_grid_spacing:
        break
    
dr_arr = [dr]
pbar = tqdm(range(n_max_iter2))
for i in pbar:
    dr /= 1.5
    r = np.arange(r_min, r_max+dr, dr)
    n_r = len(r)
    pbar.set_description(f"r_max = {r_max:.1f}; n_r = {n_r}")
    u0 = 2*r*np.exp(-r)
    E_arr, u_arr, i_max = solve_helium_kohn_sham_eq(r, u0, eps_kohn_sham)
    dr_arr.append(dr)
    E_arr_arr.append(E_arr)
    i_arr.append(i_arr)
    if abs(E_arr_arr[-1][-1]-E_arr_arr[-2][-1]) < eps_grid_spacing:
        break
    
# %%
# Task 4 plotting
E_arr = np.array([Es[-1] for Es in E_arr_arr])

# plot E progression
plt.figure(figsize=(5,4))
plt.plot(E_arr)
plt.axvline(len(r_max_arr)-1, linestyle="--", color="k", alpha=0.5)
plt.xlabel("run index")
plt.ylabel(r"$E_0 \: / \:$atomic units")
plt.show()

plt.figure(figsize=(5,4))
plt.plot(r_max_arr)
plt.xlabel("run index")
plt.ylabel(r"$r_{max}$")
plt.show()

plt.figure(figsize=(5,4))
plt.plot(np.arange(len(r_max_arr)-1,len(E_arr)), dr_arr)
plt.xlabel("run index")
plt.ylabel(r"$r_{max}$")
plt.show()
# %%
E_arr = np.array(E_arr)
abs(E_arr[-1]-E_arr[-2])# %%
plt.plot(E_arr)
print(E_arr[-1])
# %%
plt.clf()
plt.plot(r,u0/(np.sqrt(4*np.pi)*r))
for i in range(5):
    plt.plot(r, u_arr[i]/(np.sqrt(4*np.pi)*r))
# %%
