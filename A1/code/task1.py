# %% 
import numpy as np
import matplotlib.pyplot as plt
# %% 

# Functions

def spq(alpha1, alpha2):
    return np.pi**(3/2)/((alpha1+alpha2)**(3/2))

def hpq(p, q):
    hp = -4*np.pi/(p+q)**2+3*np.pi**(3/2)*p*q/(p+q)**(5/2)-4*np.pi*q/(p+q)**2
    #hp = 3*np.pi*q*np.sqrt(np.pi/(p+q)**3)-3*np.pi*q**2*np.sqrt(np.pi/(p+q)**5)-4*np.pi/(p+q)
    #hp = (np.pi**(3/2) * alpha1 *alpha2 * np.sqrt(alpha1+alpha2)- 4*alpha2**2 - 8*alpha1*alpha2 - 4*alpha1**1)/ \
    #    (alpha1**3+alpha2**3+3*alpha1**2*alpha2+3*alpha2**2*alpha1)
    #hp = -np.pi**(3/2)*(alpha2**2+5*alpha1*alpha2+4*alpha1**2-3)/ \
    #    (np.sqrt(alpha1+alpha2)*(alpha2**2+2*alpha1*alpha2+alpha1**2))
    return hp
            
def qprqs(alpha1,alpha2,alpha3,alpha4):
    qprqs = 2*np.pi**(5/2)/((alpha1+alpha3)*(alpha2*alpha4)*np.sqrt(alpha1+alpha2+alpha3+alpha4))
    return qprqs

def normalize_vec(c_vector):
    normalization = np.sum(s_matrix*c_vector.reshape((-1,1))*c_vector.reshape((1,-1)))
    #normalizationsum = c_vector @ s_matrix @ c_vector.T
    c_vector = c_vector/np.sqrt(normalization)
    return c_vector


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

# initialize  C-vector and eigenvaluevector
c_vec = np.ones((1,4))
c_vec = normalize_vec(c_vec)
eg_vec = np.array([0])

for i in range(10):
    # some broadcasting magic to get the sum 

    q_matrix_sum1 = q_matrix*c_vec.reshape(1,-1,1,1)*c_vec.reshape(1,1,1,-1)
    q_matrix_sum1=np.sum(q_matrix_sum1,axis=1)  
    q_matrix_sum1=np.sum(q_matrix_sum1,axis=2)

    #q_matrix_sum1 = np.zeros((4,4))
    #for r in range(4):
    #    for s in range(4):
    #        q_matrix_sum1 += q_matrix[:,r,:,s]*c_vec[0,r]*c_vec[0,s]
    #print(q_matrix_sum2)
    #print(q_matrix_sum1)

    # Get  matrix
    f_matrix = h_matrix + q_matrix_sum1

    # Solve generalized eigenvalue problem and obtain eigenvector

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(s_matrix)@f_matrix)
    min_index = np.argmin(eigenvalues)
    c_vec = eigenvectors[:,min_index].reshape((1,4))
    c_vec = normalize_vec(c_vec)
    #print(np.sum(s_matrix*c_vec.reshape(-1,1)*c_vec.reshape(1,-1)))
    # Get ground state energy

    a = 2*h_matrix*c_vec.reshape(-1,1)*c_vec.reshape(1,-1)
    a = np.sum(a)

    b = q_matrix*c_vec.reshape(1,-1,1,1)*c_vec.reshape(1,1,1,-1)\
                *c_vec.reshape(-1,1,1,1)*c_vec.reshape(1,1,-1,1)
    b = np.sum(b)

    eg = a+b

    eg_vec = np.append(eg_vec, eg)


print(eg_vec[-1])

# %%
# plot the energy history
plt.plot(eg_vec)

# %%
# plot the wavefunction
r = np.linspace(0,10,1000)

psi = np.sum([c_vec[0,i]*np.exp(-alphas[i]*r**2) for i in range(4)], axis=0)

plt.plot(r,psi)

# %%
