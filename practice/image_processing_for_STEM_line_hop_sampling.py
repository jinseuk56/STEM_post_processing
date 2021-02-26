
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog as tkf
import scipy.optimize as spopt
import scipy.fftpack as spfft
from sklearn.linear_model import Lasso
import glob


# In[2]:


img_directory = tkf.askdirectory()
img_adr = glob.glob(img_directory+"/*")
img_adr.sort()
print(len(img_adr))


# In[3]:


crop_size = 200
sampling_ratio = 0.5


# In[ ]:


def line_hop(img_shape, hop_bottom, hop_top):
    sampling_matrix = np.zeros(img_shape)
    print(sampling_matrix.shape)
    
    line_chosen = []
    line = 0
    hop_range = range(hop_bottom, hop_top, 1)
    while True:
        hop = np.random.choice(hop_range)
        line += hop
        if line >= img_shape[0]:
            break
        line_chosen.append(line)
    
    for i in line_chosen:
        temp = i
        for j in range(img_shape[1]):
            while(True):
                hop = np.random.choice([-1, 0, 1], p=[0.05, 0.90, 0.05])
                temp += hop
                if temp >= 0 and temp < img_shape[0]:
                    break
            sampling_matrix[temp, j] = 1
    
    
    print(("sampling ratio = %.2f %%")%(100*np.sum(sampling_matrix)/(img_shape[0]*img_shape[1])))
    
    plt.imsave(fname="sampling_matrix.png", arr=sampling_matrix, format="png", cmap="gray")
    
    return sampling_matrix


# In[4]:


def dct2d(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


# In[5]:


def CS_sequential(img_adr, crop_size, hop_limit, alphas):
    
    sy = crop_size
    sx = crop_size
    
    phi = line_hop([sy, sx], hop_limit[0], hop_limit[1])
    phi = phi.flatten()
    ri = np.where(phi==1)[0]
    
    dctrow = spfft.idct(np.identity(sx), norm='ortho', axis=0)
    dctcol = spfft.idct(np.identity(sy), norm='ortho', axis=0)
    
    A = np.kron(dctrow, dctcol)
    A = A[ri, :]
    print(A.shape)
    print("A matrix complete")
    phi = None
    dctrow = None
    dctcol = None
    
    for i in range(len(img_adr)):
        for j in range(len(alphas)):
            lasso = Lasso(alpha=alphas[j], max_iter=2000, tol=0.00001, random_state=56)
            sliced = plt.imread(img_adr[i])[:, :, 0]
            b = sliced.T.flat[ri]
            b = np.expand_dims(b, axis=1)
            print(b.shape)
            lasso.fit(A, b)
            Xa = idct2(np.array(lasso.coef_).reshape(sy, sx).T)
            plt.imsave(fname=img_adr[i][:-4]+"_alpha_%d.png"%(j), arr=Xa, format="png", cmap="gray")
            print("%d, %d"%(i, j))


# In[6]:


CS_sequential(img_adr, crop_size, hop_limit=[1, 4], alphas=[0.00001, 0.0000095, 0.0000085])

