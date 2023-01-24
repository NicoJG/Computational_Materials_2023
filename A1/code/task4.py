# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# %%
def solve_for_V_sH(r, u):
    n_r = len(r)
    dr = np.diff(r)[0]
    r_max = np.max(r)
    
    f = -u**2/r
    
    A = 1/dr**2*(np.diag(-2*np.ones(n_r), 0) \
        + np.diag(1*np.ones(n_r-1), 1) \
        + np.diag(1*np.ones(n_r-1), -1))
    
    U_0 = np.linalg.solve(A,f)
    V_sH = (U_0+r/r_max)/r
    return V_sH

def solve_kohn_sham_eq(r, V_H):
    n_r = len(r)
    dr = np.diff(r)[0]
    
    A = 1/dr**2*(np.diag(-2*np.ones(n_r), 0) \
        + np.diag(1*np.ones(n_r-1), 1) \
        + np.diag(1*np.ones(n_r-1), -1))

    M = -0.5*A + np.diag(-2/r + V_H,0) 
    
    E, u = np.linalg.eigh(M)
    E, u = E[0], u[:,0]
    
    # normalize u
    u /= -np.sqrt(np.trapz(u**2, r))
    
    return E, u

# initialize the grid
n_r = 1000
r_min = 1e-3
r_max = 10.
r = np.linspace(r_min, r_max, n_r)

# initially choose the hydrogen ground state
u0 = 2*r*np.exp(-r)
u = u0

n_iter = 10
E_arr = []
u_arr = []
for i in tqdm(range(n_iter)):
    V_sH = solve_for_V_sH(r, u)
    V_H = V_sH
    E, u = solve_kohn_sham_eq(r, V_sH)
    E0 = 2*E - 2*np.trapz(u**2*0.5*V_H, r)
    E_arr.append(E0)
    u_arr.append(u)

E_arr = np.array(E_arr)
    
# %%
plt.plot(E_arr)
# %%
plt.clf()
plt.plot(r,u0)
for i in range(5):
    plt.plot(r, u_arr[i])

# %%
