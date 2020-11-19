#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 2020

@author: A. Kofler
"""

import numpy as np
from data_processing_funcs import *

from dictionary_learning_funcs import *
from sklearn import preprocessing

"""
basic toy example for training a dictionary on a set of patches extracted
from an image and sparesly approximate the patches;
"""

#dictionary and sarse coding algoithm 
#DL must be either  ksvd, itkrm or a_itkrm_plus1
#SC must be either omp or a_omp_plus1
DL = 'a_itkrm'
SC = 'a_omp_plus1'



#K and S (have no effect if adaptive DL and SC are used)
K=128
S=8

#load the image (only one time point) and add some noise
cutoff=80
x = np.load('data/img_320.npy')
x = np.abs(x)[cutoff:320-cutoff,cutoff:320-cutoff,0]
np.random.seed(0)
x+=0.5*np.std(x)*np.random.random(x.shape)
im_size=x.shape

#extract patches
patches_size=[8,8]
strides=[1,1]
Px = img2patches(x,patches_size,strides,vectorized=True)

#center the patches
Px, mu = center_data(Px)

#choose a subset of patches for training
N_signals=5000
Ids = np.arange(Px.shape[0])
np.random.shuffle(Ids)

#get training data; normalize it
data = Px[Ids[:N_signals],...]

#initialize dictionary as  randomly chosen patches
dico = data[:K,:]
dico = preprocessing.normalize(dico,norm='l2',axis=1)  # normalise dictionary atoms
dico = np.transpose(dico) #has shape (d,K)

#train dictionary, 
n_iter_DL=100
dico, S, t_DL = learn_dico_from_data(np.transpose(data), #data must have shape (d,N)
									 DL,
									 dico, #has shape (d,K)                  
									 S=S,  #is ignored if DL=a_itkrm_plus1                 
									 K=K,  #is ignored if DL=a_itkrm_plus1
									 n_iter_DL=n_iter_DL)

#approximate the patches
Px_SC, t_SCr = sparse_approx(np.transpose(Px),dico,SC,S)

#convert to array
Px_SC = np.array(Px_SC)

#uncenter patches
Px_SC = un_center_data(Px_SC, mu)

#construct weighting matrix for reassembling the image from the patches
Wmat = construct_weight_matrix(im_size,patches_size,strides)

#re-assemble image from patches
xDL_reg = patches2img(Px_SC, im_size, patches_size, strides,Wmat = Wmat, vectorized=True)

#create figure
fig,ax=plt.subplots(1,2)
cmap=plt.cm.Greys_r
clim=[0,1000]
ax[0].imshow(x,cmap=cmap,clim=clim)
ax[1].imshow(xDL_reg,cmap=cmap,clim=clim)
#filename
if DL in ['itkrm','ksvd']:
	fname = 'denoising_example_{}_{}_S{}_K{}.pdf'.format(DL,SC,S,K)
elif DL=='a_itkrm':
	fname = 'denoising_example_aitkrm_aomp.pdf'
	
pname = 'results/denoising_example/'
fig.savefig(pname+fname)
