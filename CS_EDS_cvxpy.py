#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog as tkf
import scipy.optimize as spopt
import scipy.fftpack as spfft
import tifffile
import cvxpy as cp


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
                hop = np.random.choice([-1, 0, 1], p=[0.10, 0.80, 0.10])
                temp += hop
                if temp >= 0 and temp < img_shape[0]:
                    break
            sampling_matrix[temp, j] = 1
    
    
    print(("sampling ratio = %.2f %%")%(100*np.sum(sampling_matrix)/(img_shape[0]*img_shape[1])))
    
    return sampling_matrix


# In[ ]:


img_adr = tkf.askopenfilenames()
print(img_adr)


# In[ ]:


ox = tifffile.imread(img_adr[0])
print(ox.shape)
sr = tifffile.imread(img_adr[1])
print(sr.shape)
ti = tifffile.imread(img_adr[2])
print(ti.shape)


# In[ ]:


datatype = np.float32
print(datatype)


# In[ ]:


ch_map = ox + 2*sr + 3*ti
print(ch_map.shape)
ch_map = ch_map / np.max(ch_map)
sy, sx = ch_map.shape


# In[ ]:


plt.imshow(ch_map, cmap="Accent")


# In[ ]:


print("sampling ratio")
print(len(np.nonzero(ch_map)[0])/ch_map.size)


# In[ ]:


def dct2d(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


# In[ ]:


#phi = np.zeros([sy, sx])
#phi[np.nonzero(ch_map)] = 1.0

phi = line_hop([sy, sx], 1, 3)
phi_prime = phi.flatten()
ri = np.where(phi_prime==1)[0]


# In[ ]:


dctrow = spfft.idct(np.identity(sx, dtype=datatype), norm='ortho', axis=0)
dctcol = spfft.idct(np.identity(sy, dtype=datatype), norm='ortho', axis=0)
    
A = np.kron(dctrow, dctcol)
A = A[ri, :]
print(A.shape)
print("A matrix complete")
del phi_prime
del dctrow
del dctcol


# In[ ]:


ch_map = ch_map.astype(datatype)
A = A.astype(datatype)


# In[ ]:


b = ch_map.T.flat[ri]
print(b.shape)


# In[ ]:


x = cp.Variable(sy*sx)
print(x)


# In[ ]:


lmbd_l1 = cp.Parameter(nonneg=True)
lmbd_l1.value = 0.001


# In[ ]:


lmbd_tv = cp.Parameter(nonneg=True)
lmbd_tv.value = 0.001


# In[ ]:


constraint = [0 <= x, x <= 1]


# In[ ]:


problem = cp.Problem(cp.Minimize(cp.norm(A@x-b.ravel(), 2)
                                     +lmbd_l1*cp.norm(x, 1)+lmbd_tv*cp.tv(x)), constraint)


# In[ ]:


problem.solve(verbose=True)


# In[ ]:


tifffile.imwrite("/measurements.tif", ch_map)
tifffile.imwrite("/reconstruction.tif", x.value.reshape(sy, sx))

