# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
n_r = 1000
r_min = 0.
r_max = 10.

r = np.linspace(r_min, r_max, n_r+1)[1:]
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
plt.figure(figsize=(10,8))
plt.plot(r,psi_exact, label="psi exact")
for i in range(2):
    psi = u[:,i]/(np.sqrt(4*np.pi)*r)
    # normalize psi
    psi /= -np.sqrt(np.trapz(4*np.pi*r**2*psi**2,r))
    print(np.trapz(4*np.pi*r**2*psi**2,r))
    plt.plot(r,psi, label=f"psi numerical ({i})")
plt.xlabel(r"$r \: / \:$atomic units")
plt.ylabel(r"$\Psi \: / \:$atomic units")
plt.ylim(-0.1,0.7)
plt.grid()
plt.legend()
plt.show()

# %%
