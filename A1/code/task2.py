# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})


# %%
n_r = 1000
r_min = 0.
r_max = 10

r = np.linspace(r_min, r_max, n_r+1)[:1]

dr = np.diff(r)[0]

f = -4*r*np.exp(-2*r)

A = 1/dr**2*(np.diag(-2*np.ones(n_r), 0) \
    + np.diag(1*np.ones(n_r-1), 1) \
    + np.diag(1*np.ones(n_r-1), -1))

U_solve = np.linalg.solve(A,f)
V_H_solve = (U_solve+r/r_max)/r

V_H_exact = 1/r - (1+1/r)*np.exp(-2*r)
U_exact = V_H_exact*r - r/r_max

plt.figure(figsize=(5,4))
plt.plot(r,V_H_exact, label="Exact Hartree-potential")
plt.plot(r,V_H_solve, linestyle = 'dashed',label="Finite difference method")
plt.xlabel("Radius (a.u.)")
plt.ylabel("$V_H$ (a.u.)")
plt.title("Potential comparison ")
plt.ylim(0,1.25)
plt.grid()
plt.legend()
#plt.savefig("task2.pdf")

# %%
