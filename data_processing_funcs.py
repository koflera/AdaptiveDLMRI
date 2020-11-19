"""
Created on Wed Oct 14 

@author: A. Kofler
"""

import numpy as np

from sklearn.feature_extraction import image

from numpy.fft import fftshift, ifftshift, fftn, ifftn, fft2, ifft2, ifft, fft

import torch

from scipy import ndimage

from skimage.transform import resize

from termcolor import colored


def patches2img(Pr,target_size,patches_size,strides,Wmat=None,vectorized=False):

	"""
	function which repositions the patches contained in Pr at the correct location;
	also by properly weighting overlapping regions;

	the weighting matrix can be either pre-computed 
	(desirable if the functions has to be called several times)
	or can be computed in the function;

	"""

	if Wmat is None:
		#construct it if it nos given;;
		Wmat = construct_weight_matrix(target_size,patches_size,strides)

	#dimensionality of the image (2D or 3D)
	dim=len(target_size)

	#take the same datatye as the patches;
	dtype = Pr.dtype

	#initialize the image/volume
	x = np.zeros(target_size,dtype=dtype)
	
	if vectorized:
		#unvectorize
		#get patches dim:
		patches_dim = ()
		for pd in patches_size:
			patches_dim+=(pd,)
			
		P = np.reshape(Pr,(Pr.shape[0],) + patches_dim) #has now shape (Npatches,px,py) or (Npatches, px,py,pz)

	if dim==2:

		#get the params
		ix,iy = target_size
		sx,sy = strides
		px,py = patches_size

		#get the resulting number of patches to be processed;
		npx = np.int(np.floor((ix-px)/sx)+1)
		npy = np.int(np.floor((iy-py)/sy)+1)

		#reshape the data array Pr to be able to through the patches;
		P = np.reshape(Pr,(npx,npy,px,py))

		#reassemble;
		counter=0
		for kx in range(np.int(npx)):
			for ky in range(np.int(npy)):
				x[kx*sx:kx*sx+px,ky*sy:ky*sy+py] += P[kx,ky,:,:]
				counter+=1

	elif dim==3:

		#get the params
		ix,iy,iz = target_size
		sx,sy,sz = strides
		px,py,pz = patches_size

		npx = np.int(np.floor((ix-px)/sx)+1)
		npy = np.int(np.floor((iy-py)/sy)+1)
		npz = np.int(np.floor((iz-pz)/sz)+1)

		#reshape the data array Pr to be able to through the patches;
		P = np.reshape(Pr,(npx,npy,npz,px,py,pz))

		#reposition patches
		counter=0
		for kx in range(np.int(npx)):
			for ky in range(np.int(npy)):
				for kz in range(np.int(npz)):
					x[kx*sx:kx*sx+px,ky*sy:ky*sy+py,kz*sz:kz*sz+pz] += P[kx,ky,kz,:,:,:]

					counter+=1

	return Wmat*x

def img2patches(x,patches_size,strides,vectorized=False):

	"""
	function for extracting patches from the image x;
	x can be a 2D image (Nx,Ny) or a 3D volume (Nx,Ny,Nz)/(Nx,Ny,Nt)

	the function returns a set of (eventually normalized) patches in an array of shape

	(Np,patches_size), where Np is the resulting number of patches which could be extracted from x

	"""

	#extract all patches
	P = image.extract_patches(x,patches_size,strides)
	dim = len(x.shape)

	if dim==2: #i.e. 2D patches are considered
		px,py = patches_size
		Np = P.shape[0]*P.shape[1]
		Pr = np.reshape(P,(Np,px,py))

	elif dim==3: #i.e. 3D patches are considered
		px,py,pz = patches_size
		Np = P.shape[0]*P.shape[1]*P.shape[2]
		Pr = np.reshape(P,(Np,px,py,pz))
		
	if vectorized:
		#reshape the patches to vectors
		d = np.prod(patches_size)
		Pr = np.reshape(Pr,(Np,d))

	return Pr


def center_data(data,axis=None):
	
	"""
	subtract mean from the data
	"""
	if axis is None:

		#get the dimensionaliy of the samples;
		dim = len(data.shape[1:])

		#axes for the mean calculation (either (1,2) or (1,2,3); 
		#dimension 0 is the patches;)
		axis_vect = tuple([k for k in range(1,dim+1)])

		#calculate the mean
		mu = np.mean(data,axis=axis_vect,keepdims=1)
		
	else:
		print('TODO')

	return data-mu, mu

def un_center_data(data,mu):
	"""
	uncenter the data
	"""
	return data+mu


def construct_weight_matrix(im_shape,patches_size,strides):

	"""
	function which constructs the weighting matrix needed to weight overlapping regions
	for the process of reassembling an image or a volume from its patches;
	"""
	#get dimesionality of the image
	dim=len(im_shape)

	if dim==3:
		#define single values;
		ix,iy,iz = im_shape
		px,py,pz = patches_size
		sx,sy,sz = strides

		#weight matrix to average the image
		W = np.zeros((ix,iy,iz))

		npx = np.floor((ix-px)/sx)+1
		npy = np.floor((iy-py)/sy)+1
		npz = np.floor((iz-pz)/sz)+1

		counter=0
		for kx in range(np.int(npx)):
		
			for ky in range(np.int(npy)):
				for kz in range(np.int(npz)):
							
					W[kx*sx:kx*sx+px,ky*sy:ky*sy+py,kz*sz:kz*sz+pz] += 1
					counter+=1

	elif dim==2:
		#define single values;
		ix,iy = im_shape
		px,py = patches_size
		sx,sy = strides

		#weight matrix to average the image
		W = np.zeros((ix,iy))

		npx = np.floor((ix-px)/sx)+1
		npy = np.floor((iy-py)/sy)+1

		counter=0
		for kx in range(np.int(npx)):
			for ky in range(np.int(npy)):
				W[kx*sx:kx*sx+px,ky*sy:ky*sy+py] += 1
				counter+=1

	W[np.where(W==0)]=1
	return 1./W