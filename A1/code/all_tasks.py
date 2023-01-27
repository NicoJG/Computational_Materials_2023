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
    V_sH = np.ones_like(r)
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
plt.xlabel(r"$r \: / \:$a.u.")
plt.ylabel(r"$V_H \: / \:$a.u.")
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
plt.xlabel(r"$r \: / \:$a.u.")
plt.ylabel(r"$\Psi \: / \:$a.u.")
plt.ylim(-0.1,0.7)
plt.grid()
plt.legend()
plt.show()
plt.savefig("plots/task3.pdf")

# %%
#########################
# Task 4+5+6
#########################
# Task 4 procedure functions
def calc_E0_from_eigenvalue(E,u,r,V_H, V_xc, eps_xc):
    return 2*E - 2*np.trapz(u**2*(0.5*V_H + V_xc - eps_xc), r)

# Task 5
def calc_exchange_potential(r,u):
    n =  u**2/(2*np.pi)
    eps_x = -3/4*(3*n/np.pi)**(1/3)
    V_x = (3*n/np.pi)**(1/3)
    V_x = eps_x + n*np.gradient(eps_x,n)
    return eps_x, V_x

# Task 6
def calc_correlation_potential(r,u):
    eps_c = np.zeros_like(r)
    n =  u**2/(2*np.pi)
    r_s = (3/(4*np.pi*n))**(1/3)

    A = 0.0311
    B  = -0.048
    C = 0.0020
    D = -0.116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334

    mask = r_s >= 1
    eps_c[mask] = gamma/(1+beta1*np.sqrt(r_s[mask])+beta2*r_s[mask])
    eps_c[~mask] = A*np.log(r_s[~mask]) + B + C*r_s[~mask]*np.log(r_s[~mask]) + D*r_s[~mask]
    
    V_c = eps_c + n*np.gradient(eps_c,n)
    
    return eps_c, V_c

def solve_helium_kohn_sham_eq(r, u0, eps, n_max_iter=100,
                    include_exchange=False, include_correlation=False):
    u = u0
    E_arr = []
    E0_arr = []
    u_arr = []
    for i in tqdm(range(n_max_iter),leave=False):
        V_sH = solve_for_V_sH(r, u)

        # for task 5 and 6:
        if include_exchange or include_correlation:
            V_H = 2*V_sH
        else:
            V_H = V_sH
        if include_exchange:
            eps_x, V_x = calc_exchange_potential(r,u)
        else:
            eps_x, V_x = 0, 0
        if include_correlation:
            eps_c, V_c = calc_correlation_potential(r,u)
        else:
            eps_c, V_c = 0, 0
        V_xc = V_x + V_c
        eps_xc = eps_x + eps_c

        potential = -2/r + V_H + V_xc

        E, u = solve_kohn_sham_eq(r, potential)
        E0 = calc_E0_from_eigenvalue(E,u,r,V_H, V_xc, eps_xc)
        E_arr.append(E)
        E0_arr.append(E0)
        u_arr.append(u)
        if len(E_arr)>1:
            if abs(E_arr[-1]-E_arr[-2])*hartree_energy < eps:
                return E_arr, E0_arr, u_arr, i

    print("WARNING: n_max_iter is too low!")
    return E_arr, E0_arr, u_arr, i

# %%
# Task 4 calculations
eps_kohn_sham = 1e-5 * hartree_energy # eV
eps_r_max = 1e-5 * hartree_energy
eps_dr = 1e-2
n_max_iter1 = 20
n_max_iter2 = 20

# initial grid
dr = 0.01
r_min = 0.
r_max = 2.
r = np.arange(r_min, r_max+dr, dr) # so that r_max is included

include_exchange = True
include_correlation = True

# initially choose the hydrogen ground state
u0 = 2*r*np.exp(-r)

E_arr, E0_arr, u_arr, i_max = solve_helium_kohn_sham_eq(r,u0, eps_kohn_sham, 
                                include_exchange=include_exchange, 
                                include_correlation=include_correlation)
r_max_arr = [r_max]
E_arr_arr = [E_arr]
E0_arr_arr = [E0_arr]
i_arr = [i_max]


pbar = tqdm(range(n_max_iter1))
for i in pbar:
    r_max *= 1.5
    r = np.arange(r_min, r_max+dr, dr)
    n_r = len(r)
    pbar.set_description(f"r_max = {r_max:.1f}; n_r = {n_r}")
    u0 = 2*r*np.exp(-r)
    E_arr, E0_arr, u_arr, i_max = solve_helium_kohn_sham_eq(r, u0, eps_kohn_sham, 
                                include_exchange=include_exchange, 
                                include_correlation=include_correlation)
    r_max_arr.append(r_max)
    E_arr_arr.append(E_arr)
    E0_arr_arr.append(E0_arr)
    i_arr.append(i_max)
    if abs(E0_arr_arr[-1][-1]-E0_arr_arr[-2][-1]) < eps_r_max:
        break
    
dr_arr = [dr]
pbar = tqdm(range(n_max_iter2))
for i in pbar:
    dr *= 0.75
    r = np.arange(r_min, r_max+dr, dr)
    n_r = len(r)
    pbar.set_description(f"dr = {dr:.5f}; n_r = {n_r}")
    u0 = 2*r*np.exp(-r)
    E_arr, E0_arr, u_arr, i_max = solve_helium_kohn_sham_eq(r, u0, eps_kohn_sham, 
                                include_exchange=include_exchange, 
                                include_correlation=include_correlation)
    dr_arr.append(dr)
    E_arr_arr.append(E_arr)
    E0_arr_arr.append(E0_arr)
    i_arr.append(i_arr)
    if abs(E0_arr_arr[-1][-1]-E0_arr_arr[-2][-1]) < eps_dr:
        break
    
# %%
# Task 4 plotting
E_arr = np.array([Es[-1] for Es in E_arr_arr])
E0_arr = np.array([Es[-1] for Es in E0_arr_arr])

# plot E progression
plt.figure(figsize=(5,4))
plt.plot(E0_arr)
plt.axvline(len(r_max_arr)-1, linestyle=":", color="k", alpha=0.5, label="$r_{max}$ is converged")
plt.axhline(E0_arr[-1], linestyle="--", color="k", alpha=0.5, label=f"$E_0 = {E0_arr[-1]:.5f} \\: / \\:$a.u.")
plt.xlabel("run index")
plt.ylabel(r"$E_0 \: / \:$a.u.")
plt.legend()
plt.show()

# plot E progression
plt.figure(figsize=(5,4))
plt.plot(E_arr)
plt.axvline(len(r_max_arr)-1, linestyle=":", color="k", alpha=0.5, label="$r_{max}$ is converged")
plt.axhline(E_arr[-1], linestyle="--", color="k", alpha=0.5, label=f"$E = {E_arr[-1]:.5f} \\: / \\:$a.u.")
plt.xlabel("run index")
plt.ylabel(r"$E \: / \:$a.u.")
plt.legend()
plt.show()

plt.figure(figsize=(5,4))
plt.plot(r_max_arr)
plt.xlabel("run index")
plt.ylabel(r"$r_{max} \: / \:$a.u.")
plt.show()

plt.figure(figsize=(5,4))
plt.plot(np.arange(len(r_max_arr)-1,len(E_arr)), dr_arr)
plt.xlabel("run index")
plt.ylabel(r"$\Delta r \: / \:$a.u.")
plt.show()

# plot the final wavefunction
u = u_arr[-1]
psi = calc_psi_from_u(r,u)

plt.figure(figsize=(5,4))
plt.plot(r,psi)
plt.xlabel(r"$r \: / \:$a.u.")
plt.ylabel(r"$\Psi \: / \:$a.u.")
plt.show()

# %%
