# %% 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

# %% 

# Functions

def xi(alpha, radie):
    return np.exp(-alpha*radie**2)

def wavefunction(alpha_vector, c_vector, radie):
    wavesum = 0
    for i in range(len(alpha_vector)):
        wavesum += c_vector[0][i]*xi(alpha_vector[i],radie)
        print(c_vector[0][i])
        print(alpha_vector[i])
    return wavesum 

def spq(alpha1, alpha2):
    return np.pi**(3/2)/((alpha1+alpha2)**(3/2))

def hpq(p, q):
    hp = -4*np.pi/(p+q)+3*np.pi**(3/2)*p*q/(p+q)**(5/2)
    return hp
            
def qprqs(alpha1,alpha2,alpha3,alpha4):
    qprqs = 2*np.pi**(5/2)/((alpha1+alpha3)*(alpha2+alpha4)*np.sqrt(alpha1+alpha2+alpha3+alpha4))
    return qprqs

def normalize_vec(vector):
    normalizationsum = vector @ s_matrix @ vector.T
    vector = vector/np.sqrt(normalizationsum)
    return vector


# Optimal values 

a_1 = 0.297104
a_2 = 1.236745
a_3 = 5.749982
a_4 = 38.216677
alphas = [a_1, a_2, a_3, a_4]



# Construct matrices

s_matrix = np.ones((4,4))
for i in range(len(s_matrix)):
    for j in range(len(s_matrix)):
        s_matrix[i,j] = spq(alphas[i],alphas[j])

h_matrix = np.ones((4,4))
for i in range(len(h_matrix)):
    for j in range(len(h_matrix)):
        h_matrix[i,j] = hpq(alphas[i],alphas[j])

q_matrix = np.ones((4,4,4,4))
for i in range(len(q_matrix)):
    for j in range(len(q_matrix)):
        for k in range(len(q_matrix)):
            for l in range(len(q_matrix)):
                q_matrix[i,j,k,l] = qprqs(alphas[i],alphas[j],alphas[k], alphas[l])
print("Q")
print(q_matrix[1][3])


# initialize  C-vector, eigenvaluevector and difference
c_vec = np.ones((1,4))
c_vec=normalize_vec(c_vec)
eg_vec = np.array([0])
eg_diff = 1
eg_old = 1
i=0

while eg_diff > 10**(-5): # as in the excercise

#for i in range(50):

    
    # some broadcasting magic to get the sum 

    q_matrix_sum1 = q_matrix*c_vec.reshape(1,-1,1,1)*c_vec.reshape(1,1,1,-1)
    q_matrix_sum1=np.sum(q_matrix_sum1,axis=1)  
    q_matrix_sum1=np.sum(q_matrix_sum1,axis=2)

    # Get  matrix
    f_matrix = h_matrix + q_matrix_sum1

    # Solve generalized eigenvalue problem and obtain eigenvector
    
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(s_matrix)@f_matrix)
    min_index = np.argmin(eigenvalues)
    c_vec = eigenvectors[:,min_index].reshape((1,4))
    c_vec=normalize_vec(c_vec)
   
    # Get ground state energy

    a = 2*h_matrix*c_vec.reshape(-1,1)*c_vec.reshape(1,-1)
    a = np.sum(a)
    b = q_matrix*c_vec.reshape(1,-1,1,1)*c_vec.reshape(1,1,1,-1)\
                *c_vec.reshape(-1,1,1,1)*c_vec.reshape(1,1,-1,1)
    b = np.sum(b)
    eg = a+b
    eg_vec = np.append(eg_vec, eg)
    
    print(i)
    i += 1
    eg_diff = np.abs(eg-eg_old)
    eg_old = eg
    print(eg_diff)


print(eg_vec[-1])

# %% 
# Plot regular

# Eg plot and wavefunction for boolean
fig, ax = plt.subplots()
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Energy (a.u.)', fontsize=14)
ax.set_title('Convergence for 5 iterations',fontsize=16)
ax.grid()
ax.legend()
plt.plot(eg_vec)
#fig.savefig('convergence_5it.pdf')

# Wavefunction
radie = np.linspace(0,4,1000)
wavefunc = wavefunction(alphas, c_vec, radie)*-1

fig, ax = plt.subplots()
ax.set_xlabel('Radius (a.u)', fontsize=14)
ax.set_ylabel('$\Psi(r)$ (a.u)', fontsize=14)
ax.set_title('$\Psi(r)$ for 5 iterations',fontsize=16)
ax.grid()
ax.legend()
plt.plot(radie, wavefunc)
#fig.savefig('wavefunc_5it.pdf')

# %% Foor loop variant

# Optimal values 

a_1 = 0.297104
a_2 = 1.236745
a_3 = 5.749982
a_4 = 38.216677
alphas = [a_1, a_2, a_3, a_4]



# Construct matrices

s_matrix = np.ones((4,4))
for i in range(len(s_matrix)):
    for j in range(len(s_matrix)):
        s_matrix[i,j] = spq(alphas[i],alphas[j])

h_matrix = np.ones((4,4))
for i in range(len(h_matrix)):
    for j in range(len(h_matrix)):
        h_matrix[i,j] = hpq(alphas[i],alphas[j])

q_matrix = np.ones((4,4,4,4))
for i in range(len(q_matrix)):
    for j in range(len(q_matrix)):
        for k in range(len(q_matrix)):
            for l in range(len(q_matrix)):
                q_matrix[i,j,k,l] = qprqs(alphas[i],alphas[j],alphas[k], alphas[l])
print("Q")
print(q_matrix[1][3])


# initialize  C-vector, eigenvaluevector and difference
c_vec50 = np.ones((1,4))
c_vec50=normalize_vec(c_vec50)
eg_vec50 = np.array([0])
eg_diff = 1
eg_old = 1
i=0

#while eg_diff > 10**(-5): # as in the excercise

for i in range(50):

    
    # some broadcasting magic to get the sum 

    q_matrix_sum1 = q_matrix*c_vec50.reshape(1,-1,1,1)*c_vec50.reshape(1,1,1,-1)
    q_matrix_sum1=np.sum(q_matrix_sum1,axis=1)  
    q_matrix_sum1=np.sum(q_matrix_sum1,axis=2)

    # Get  matrix
    f_matrix = h_matrix + q_matrix_sum1

    # Solve generalized eigenvalue problem and obtain eigenvector
    
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(s_matrix)@f_matrix)
    min_index = np.argmin(eigenvalues)
    c_vec50 = eigenvectors[:,min_index].reshape((1,4))
    c_vec50=normalize_vec(c_vec50)
   
    # Get ground state energy

    a = 2*h_matrix*c_vec50.reshape(-1,1)*c_vec50.reshape(1,-1)
    a = np.sum(a)
    b = q_matrix*c_vec50.reshape(1,-1,1,1)*c_vec50.reshape(1,1,1,-1)\
                *c_vec50.reshape(-1,1,1,1)*c_vec50.reshape(1,1,-1,1)
    b = np.sum(b)
    eg = a+b
    eg_vec50 = np.append(eg_vec50, eg)
    
    print(i)
    i += 1
    eg_diff = np.abs(eg-eg_old)
    eg_old = eg
    print(eg_diff)


print(eg_vec50[-1])

# %%
# Convergence and wavefunction for 50 iterations

fig, ax = plt.subplots()
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Energy (a.u.)', fontsize=14)
ax.set_title('Convergence for 50 iterations',fontsize=16)
ax.grid()
ax.legend()
plt.plot(eg_vec50)
#fig.savefig('convergence_50it.pdf')

# Wavefunction
radie = np.linspace(0,4,1000)
wavefunc50 = wavefunction(alphas, c_vec50, radie)*-1

fig, ax = plt.subplots()
ax.set_xlabel('Radius (a.u.)', fontsize=14)
ax.set_ylabel('$\Psi(r)$ (a.u.)', fontsize=14)
ax.set_title('$\Psi(r)$ for different iterations',fontsize=16)
ax.grid()

plt.plot(radie, wavefunc, label = '5 iterations')
plt.plot(radie, wavefunc50, linestyle = 'dashed', label = '50 iterations')
ax.legend()
fig.savefig('wavefunc_comp.pdf')

# %%
