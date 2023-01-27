# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})
# %%
n_r = 1000
r_min = 1e-3
r_max = 10.

r = np.linspace(r_min, r_max, n_r)
dr = np.diff(r)[0]

A = 1/dr**2*(np.diag(-2*np.ones(n_r), 0) \
    + np.diag(1*np.ones(n_r-1), 1) \
    + np.diag(1*np.ones(n_r-1), -1))

M = -0.5*A - np.diag(1/r,0) 

# %%
E, u = np.linalg.eigh(M)


psi_exact = 1/np.sqrt(np.pi)*np.exp(-r)
print(np.trapz(4*np.pi*r**2*psi_exact**2,r))

# %%
#plt.figure(figsize=(10,8))
plt.plot(r,psi_exact, label="$\Psi_{exact}$")
for i in range(2):
    psi = u[:,i]/(np.sqrt(4*np.pi)*r)
    # normalize psi
    psi /= -np.sqrt(np.trapz(4*np.pi*r**2*psi**2,r))
    print(np.trapz(4*np.pi*r**2*psi**2,r))
    plt.plot(r,psi, label=f"Numerical $\Psi_{i}$  ")
plt.xlabel("Radius (a.u.)")
plt.ylabel("$\Psi(r)$ (a.u.)")
plt.title("Comparison of numerical wavefucntion for hydrogen")
plt.ylim(-0.1,0.7)
plt.grid()
plt.legend()
plt.savefig("task3_both.pdf")
plt.show()

# %%
