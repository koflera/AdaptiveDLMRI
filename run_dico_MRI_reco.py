"""
Created on Wed Oct 14 

@author: A. Kofler

"""

import numpy as np

import torch

import matplotlib.pyplot as plt
plt.ioff()

import matplotlib.gridspec as gridspec

from sklearn import preprocessing

from nufft_operator import Dyn2DRadEncObj
from operator_helper_funcs import ConjGrad, HOperator, add_gaussian_noise, np2torch, torch2np

from data_processing_funcs import img2patches, patches2img, center_data, un_center_data, construct_weight_matrix

from dictionary_learning_funcs import learn_dico_from_data, sparse_approx

import os

from termcolor import colored

import time

import argparse
parser = argparse.ArgumentParser()				
parser.add_argument('--sigma',default=1e-2,type=float)		 	    #noise level in the k-space data
parser.add_argument('--Psi_training',default=0,type=int)		    #wheter to train the dictionary during the reconstruction or not;
parser.add_argument('--npcg',default=4,type=int)	 			    #number of PCG-updates for updaing the solution estimate	
parser.add_argument('--N',default=20000,type=int)	 			    #number of patches used for training the dictionary online
parser.add_argument('--n_DL',default=20,type=int)				    #number of iterations of the respective DL algorthm			
parser.add_argument('--T',default=6,type=int)	 				    #number overall iterations					
parser.add_argument('--lambda_reg',default=1.,type=float)  		  	#the regularization parameter 
parser.add_argument('--DL',default='a_itkrm',type=str)			  	#DL-algorithm - either "kvd", "itkrm" or "a_itkrm"								
parser.add_argument('--SC',default='a_omp_plus1',type=str)		  	#SC-algorhm - either "omp" or "a_omp_plus1"
parser.add_argument('--K',default=64,type=int)		 			  	#number of atoms of the dictionary; only has an effect for kvd and itkrm								
parser.add_argument('--S',default=8,type=int)	 				   	#sparsity level; only has an aeffect for ksvd and itkrm
parser.add_argument('--mumax',default=0.7,type=float)			    #coherence mumax for a_itkrm;
parser.add_argument('--minobs_str',default='d',type=str)		    #minimal number of observations M for aitkrm
parser.add_argument('--use_GPU',default=1,type=int)		 	 	 	#minimal number of observations M for aitkrm
args = parser.parse_args()

""""

Minimal Example Code for our paper

"Adaptive Sparsity Level and Dictionary Size Estimation for
Image Reconstruction in Accelerated 2D Radial Cine MRI"

"

by M. Pali et al., Medical Physics 2020

We consider the two reconstruction problems given by

min_{x,D,{\gamma_j}_j} ||Ex -y||_2^2 + \sum_j ||R_j x - D\gamma_j||_2^2 + ||\gamma_j||_0, (P1)

and 

min_{x,{\gamma_j}_j} ||Ex -y||_2^2 + \sum_j ||R_j x - D\gamma_j||_2^2 + ||\gamma_j||_0, (P2)


The solution is obtained by alternatively updating the dictionary and the 
sparse codes and the reconstruction estimate for (P1) and only the sparse codes 
and the reconstruction update for (P2), i.e. D is fixed

We provide the reconstruction code for a 2D cine MRI image with Nt=20 cardiach phases.

In order to keep the example as simple as possible, the k-space trajectories,
the density compensation functions and the coil-sensitivity maps were previously
calculated.

We provide an implementation of the 2D radial non-uniform FFT (NUFFT) encoding-operator.
In this example, the NUFFT operator samples the image of shape (320,320,20)
along pre-defined k-space trajectories.
For details about the library Torch KB-NUFFT please see
https://github.com/mmuckley/torchkbnufft

The density compensation function was previously calculated using the 
voronoi-neighbourhodd of each k-spade point.

"""

"""
Define parameters to create folder for results and create data;
"""
#define the image size
im_size = [320,320,20]
Nx,Ny,Nt = im_size

#whether to use a GPU for the radial operator
use_GPU = args.use_GPU

#the reconstructin parameters;
T = args.T
strides=[2,2,2]
patches_size = [4,4,4]
lambda_reg = args.lambda_reg
npcg=args.npcg
K = args.K
S = args.S
DL = args.DL
SC = args.SC
Psi_training = args.Psi_training
N = args.N
nDL = args.n_DL

if DL =='a_itkrm':
	K_str = ''
	S_str = ''
elif DL in ['ksvd','itkrm']:
	K_str = '_K{}'.format(K)
	S_str = '_S{}'.format(S)
	
#create folder;
experiment_folder = '{}_{}{}{}/'.format(DL,SC,K_str,S_str)
if Psi_training:
	Psi_str = 'Psi_Training'
else:
	Psi_str = 'Psi_Fixed'
	
cwd = os.getcwd()

#create the path if does not exist
if not os.path.exists(cwd+'/results/MRI_reco'):
	os.mkdir(cwd+'/results/MRI_reco')
	
results_folder = cwd+'/results/MRI_reco/{}/'.format(Psi_str) + experiment_folder
dico_folder = results_folder  + '/dico_folder/'
if not os.path.exists(results_folder):
	
	os.mkdir(results_folder)
	os.mkdir(dico_folder)
	
#define the phantom
np.random.seed(0)

#load ground-truth immage
xf  = np.load('data/img_320.npy')

#convert to tensor
xf_tensor = np2torch(xf, use_GPU = use_GPU)

#load k-space trajectories of shape (Nrad,20)
ktraj = np.load('data/ktraj_320.npy') 

#convert to tensor of shape (1,2,Nrad,20)
ktraj_tensor =  torch.stack([torch.tensor(ktraj.real),torch.tensor(ktraj.imag)],dim=0).unsqueeze(0)

#load complex-valued coil-sensitivy maps of shape (12,320,320)
csmap = np.load('data/csmap_320.npy')  

#convert to tensor of shape (1,12,2,320,320)
csmap_tensor = torch.stack([torch.tensor(csmap.real),torch.tensor(csmap.imag)],dim=1).unsqueeze(0) 

#load density compensation function
dcomp = np.load('data/dcomp_320.npy')  #shape (Nrad,20)

#convert to tensor of shape (1,1,1,Nrad,20)
dcomp_tensor = torch.tensor(dcomp).unsqueeze(0).unsqueeze(0).unsqueeze(0) 

if use_GPU:
	ktraj_tensor = ktraj_tensor.cuda()
	csmap_tensor = csmap_tensor.cuda()
	dcomp_tensor = dcomp_tensor.cuda()

#define the encoding operator object
print(colored('Define the Objects;','green'))
EncObject = Dyn2DRadEncObj(im_size,ktraj_tensor,dcomp_tensor,csmap_tensor)

if use_GPU:
	EncObject = EncObject.cuda()

print(colored('apply the forward operator ;','green'))
np.random.seed(0)
sigma=args.sigma

ku_data = EncObject.apply_A(xf_tensor)
ku_data = add_gaussian_noise(ku_data,sigma=sigma)

print(colored('apply the adjoint operator ;','green'))
xu_tensor = EncObject.apply_Adag(ku_data)

#convert to numpy array
xu = torch2np(xu_tensor, use_GPU = use_GPU)

#define the H operator which is needed for the reconstruction update;
HOp = HOperator(EncObject.apply_AdagA_Toeplitz, lambda_reg)

#initialize conjugate gradient object
CG = ConjGrad()

#initialize first estimate of the solution;
xk_tensor = xu_tensor.clone()

#the weight matrix
Wmat = construct_weight_matrix(im_size,patches_size,strides)

#time counters
t_SC_total = 0
t_DL_total = 0
t_PCG_total = 0

#where to store S and K
if DL=='a_itkrm':
	S_vect = [] 
	K_vect = []

for admm_iter in range(T):

	###########################################################
	# block 1 (U1) --> Learn the dictionary on image-patches;
	###########################################################
	
	#move to cpu for extracting patches and centering the data;
	xk = torch2np(xk_tensor, use_GPU = use_GPU)
	
	#i) pre-process initial image: extract patches, center and reshape it;
	Pxkr = img2patches(np.real(xk),patches_size,strides,vectorized=True)
	Pxki = img2patches(np.imag(xk),patches_size,strides,vectorized=True)
			
	#if desired, learn the dictionary as well during the reconstruction;
	if Psi_training:
		
		#concatenate the both;
		Pxk = np.concatenate([Pxkr,Pxki],axis=0)
		
		#center the patches
		Pxk, mu = center_data(Pxk)
		
		#how many signals to use for training the dictionary oline;
		N_signals = N
		n_iter_DL = nDL
		
		#indices of all possible patches; shuffle them
		Ids = np.arange(Pxk.shape[0])
		np.random.shuffle(Ids)
		
		#get training data; normalize it
		data = Pxk[Ids[:N_signals],...]
		
		if admm_iter==0:
			#initialize it as random signals;
			dico0 = data[:K,:]
			dico0 = preprocessing.normalize(dico0,norm='l2',axis=1)  # normalise dictionary atoms
			dico0 = np.transpose(dico0) #has shape (d,K)
		else:
			#load the one previously 
			dico0 = np.load(dico_folder+'dico{}.npy'.format(admm_iter-1)) #already has shape (d,K)
		
		#learn the dictionary;
		dico, S, t_DL = learn_dico_from_data(np.transpose(data), #data has shape (d,N)
						 DL,
						 dico0, #has shape (d,K)                  
						 S=S,                   
						 K=K,
						 n_iter_DL=n_iter_DL)
		
		#save the dictionary;
		np.save(dico_folder+'dico{}'.format(admm_iter),dico)
		t_DL_total+=t_DL
		
		if DL=='a_itkrm':
				
			#track estimated S and estimated  K
			S_vect.append(S)
			K_vect.append(dico.shape[1])
			
	else:
		#use a pre-trained dictionary;
		if DL in ['ksvd','itkrm']:
			dico = np.load(cwd+'/pre_trained_dicos/{}/dico_K{}_S{}.npy'.format(DL,K,S))
		elif DL in ['a_itkrm']:
			dico = np.load(cwd+'/pre_trained_dicos/a_itkrm/dico.npy')
		
			
	###########################################################
	# block 2 (U2) --> sparse-coding of the patches;
	###########################################################
	
	#center and normalize data
	Pxkr, mur = center_data(Pxkr)
	Pxki, mui = center_data(Pxki)
	
	#do sparse coding; data is (N,d) --> transpose it to (d,N) for our purpposes
	print(colored('sparse coding using; real part','green'))
	Pxkr, t_SCr = sparse_approx(np.transpose(Pxkr),dico,SC,S) #N.B. is np.matrix -> convert it to a np.array
	Pxkr = np.array(Pxkr)
	print(colored('sparse coding using; imaginary part','green'))
	Pxki,t_SCi = sparse_approx(np.transpose(Pxki),dico,SC,S)
	Pxki = np.array(Pxki)
		
	#the sum of the time needed for sparse coding;
	t_SC = t_SCr + t_SCi
	t_SC_total+=t_SC
			
	#uncenter the data
	Pxkr = un_center_data(Pxkr, mur)
	Pxki = un_center_data(Pxki, mui)
	
	#get the complex-valued patches;
	Pxk = Pxkr+1j*Pxki

	#reassemble image from patches;
	xDL_reg = patches2img(Pxk, im_size, patches_size, strides,Wmat = Wmat, vectorized=True)
		
	#convert to pytorch
	xDL_reg_tensor = np2torch(xDL_reg, use_GPU = use_GPU)
	
	###########################################################
	# block 3--> solve the linear system;
	###########################################################
	
	#now use xDL_reg as regularizer an consider the classical functional and solve Hx=b;
	rhs = xu_tensor + lambda_reg*xDL_reg_tensor
	
	#perform CG;
	print(colored('update image estimate using CG','green'))
	t0_PCG = time.time()
	
	#solve the system 
	xk_tensor = CG(HOp, xk_tensor, rhs, niter = npcg )
	
	#track time
	t1_PCG = time.time()-t0_PCG
	t_PCG_total+=t1_PCG
	

#save the times
if Psi_training: #only available if DL was used 
	np.save(results_folder+'t_DL.npy',t_DL_total)

np.save(results_folder+'t_SC.npy',t_SC_total)
np.save(results_folder+'t_PCG.npy',t_PCG_total)

#save sparsty level and dictionary sizes over iteration
if DL=='a_itkrm':
	np.save(results_folder+'S_vect.npy',S_vect)
	np.save(results_folder+'K_vect.npy',K_vect)

#save the solution; #convert it to numpy-array
xDL_reg = torch2np(xk_tensor, use_GPU = use_GPU)

#save the final reconstruction
np.save(results_folder+'xDL_reg.npy',xDL_reg)

#create figure
fig = plt.figure(figsize=(6*5,4*5))
gs = gridspec.GridSpec(2,6, 
					   hspace = 0.05, wspace = 0.1, height_ratios= [1,1], width_ratios = [1,8,1,8,1,8])
cutoff = 80
kt=15
kx=52
Nx,Ny,Nt = im_size	
arrs_list = [xu, xDL_reg, xf]
errs_list = [np.abs(xu-xf), np.abs(xDL_reg-xf), np.zeros(xf.shape)]
clim=[0,1000]
ks=0
font_size=20
clim_img = plt.cm.Greys_r
clim_err = plt.cm.viridis
arr_titles = np.array([
	['yt-view',  'NUFFT - xy-view'],
			  ['yt-view', 'DL-reg - xy-view'],
			  ['yt-view', 'GT - xy-view'],		  
			  ])
	  		  
for k in range(3):
	 fig.add_subplot(gs[0,ks])
	 plt.imshow(np.abs(arrs_list[k][cutoff:Nx-cutoff,kx,:]),cmap=clim_img,clim=clim)
	 plt.xticks([])
	 plt.yticks([])
	 plt.title(arr_titles[k,0],fontsize=font_size)
	 
	 fig.add_subplot(gs[0,ks+1])
	 plt.imshow(np.abs(arrs_list[k][cutoff:Nx-cutoff,cutoff:Ny-cutoff,kt]),cmap=clim_img,clim=clim)
	 plt.xticks([])
	 plt.yticks([])
	 plt.title(arr_titles[k,1],fontsize=font_size)
	 
	 #the respective errors
	 fig.add_subplot(gs[1,ks])
	 plt.imshow(3*np.abs(errs_list[k][cutoff:Nx-cutoff,kx,:]),cmap=clim_err,clim=clim)
	 plt.xticks([])
	 plt.yticks([])

	 fig.add_subplot(gs[1,ks+1])
	 plt.imshow(3*np.abs(errs_list[k][cutoff:Nx-cutoff,cutoff:Ny-cutoff,kt]),cmap=clim_err,clim=clim)
	 plt.xticks([])
	 plt.yticks([])

	 ks+=2
	 
if DL in ['ksvd','itkrm']:
	fig.savefig(results_folder+'DL_reco_{}_{}_K{}_S{}_{}.pdf'.format(DL,SC,K,S,Psi_str),layout='tight',pad_inches=0)
elif DL in ['a_itkrm']:
	fig.savefig(results_folder+'DL_reco_aitkrm_aomp_{}.pdf'.format(Psi_str),layout='tight',pad_inches=0)
plt.close('all')
	 
 	
