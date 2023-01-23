# %% 
import numpy as np

# %% 

# Functions

def xi(alpha, radius):  
    return np.exp(-alpha*radius)

def spq(alpha1, alpha2):
    return np.pi**(3/2)/((alpha1+alpha2)**(3/2))

def hpq(alpha1, alpha2):
    hp = -np.pi**(3/2)*(alpha2**2+5*alpha1*alpha2+4*alpha1**2-3)/ \
        (np.sqrt(alpha1+alpha2)*(alpha2**2+2*alpha1*alpha2+alpha1**2))
    return hp
            
def qprqs(alpha1,alpha2,alpha3,alpha4):
    qprqs = 2*np.pi**(5/2)/((alpha1+alpha3)*(alpha2*alpha4)*\
            np.sqrt(alpha1+alpha2+alpha3+alpha4))
    return qprqs

def normalize_vec(vector):
    normalizationsum = c_vec @ s_matrix @ c_vec.T
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

# initial and normalize C-vector
c_vec = np.ones((1,4))
c_vec=normalize_vec(c_vec)


f_matrix = h_matrix # Add summation of big element matrix


# %%
