# %% 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

# %% 

# Functions

def spq(alpha1, alpha2):
    return np.pi**(3/2)/((alpha1+alpha2)**(3/2))

def hpq(p, q):
    hp = -4*np.pi/(p+q)+3*np.pi**(3/2)*p*q/(p+q)**(5/2)

    #hp = 3*np.pi*q*np.sqrt(np.pi/(p+q)**3)-3*np.pi*q**2*np.sqrt(np.pi/(p+q)**5)-4*np.pi/(p+q)
    #hp = (np.pi**(3/2) * p *q * np.sqrt(p+q)- 4*q**2 - 8*p*q - 4*p**1)/ \
    #   (p**3+q**3+3*p**2*q+3*q**2*p)
    #hp = -np.pi**(3/2)*(alpha2**2+5*alpha1*alpha2+4*alpha1**2-3)/ \
    #   (np.sqrt(alpha1+alpha2)*(alpha2**2+2*alpha1*alpha2+alpha1**2))
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
while eg_diff > 10**(-5):

    
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

# Eg plot and wavefunction for boolean
fig, ax = plt.subplots()
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Energy (a.u.)', fontsize=14)
ax.set_title('Convergence for 5 iterations',fontsize=16)
ax.grid()
ax.legend()
plt.plot(eg_vec)
#fig.savefig('convergence_5it.pdf')
fig, ax = plt.subplots()
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Energy (a.u.)', fontsize=14)
ax.set_title('Convergence for 5 iterations',fontsize=16)
ax.grid()
ax.legend()
plt.plot(eg_vec)

# %%
