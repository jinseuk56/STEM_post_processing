
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog as tkf
import scipy.optimize as spopt
import scipy.fftpack as spfft
from sklearn.linear_model import Lasso
import glob


# In[ ]:


img_directory = tkf.askdirectory()
img_adr = glob.glob(img_directory+"/*")
img_adr.sort()
print(len(img_adr))


# In[ ]:


crop_size = 200
sampling_ratio = 0.5


# In[ ]:


def dct2d(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


# In[ ]:


def CS_sequential(img_adr, crop_size, s_r, alphas):
    
    sy = crop_size
    sx = crop_size
    
    k = round(sy*sx*s_r)
    ri = np.random.choice(sy*sx, k, replace=False)
    
    dctrow = spfft.idct(np.identity(sx), norm='ortho', axis=0)
    dctcol = spfft.idct(np.identity(sy), norm='ortho', axis=0)
    
    A = np.kron(dctrow, dctcol)
    A = A[ri, :]
    print(A.shape)
    print("A matrix complete")
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


# In[ ]:


CS_sequential(img_adr, crop_size, sampling_ratio, alphas=[0.00001, 0.0000095, 0.0000085])

