import torch
import torch.nn as nn

import numpy as np

def torch2np(x, use_GPU=0):
	
	"""
	convert a torch cine MR image (1,1,2,Nx,Ny,Nt) 
	to a numpy array (Nx,Ny,Nt)
	
	if a GPU is used, first move the tensor to the CPU
	
	"""
	
	if use_GPU:
		x = x.cpu()
		
	x = x.squeeze(0).squeeze(0).numpy()
	x = x[0,...] + 1j*x[1,...]
	
	return x
	
	
def np2torch(x, use_GPU=0):
	
	"""
	convert a cine MR image (Nx,Ny,Nt) to a torch tensor (1,1,2,Nx,Ny,Nt);
	
	if a GPU is used, further move the tensor to the GPU 
	
	"""
	x = torch.stack([torch.tensor(np.real(x)),torch.tensor(np.imag(x))],dim=0)
	x = x.unsqueeze(0).unsqueeze(0)
	
	if use_GPU:
		x = x.cuda()
		
	return x
	

def add_gaussian_noise(kdata,sigma=0.02):
	
	"""
	function for adding normally-distributed noise to the measured
	k-space data.
													
	"""
	np.random.seed(0)
	
	#the deivice on which the k-space data is located;
	device = kdata.device
	sigma= torch.tensor(sigma).to(device)
	
	mb, Nc, n_ch, Nrad, Nt = kdata.shape 
		
	#center the data and add normally distributed noise
	for kc in range(Nc):
		for kt in range(Nt):
			mu, std = torch.mean(kdata[:,kc,:,:,kt]), torch.std(kdata[:,kc,:,:,kt])
			
			kdata[:,kc,:,:,kt]-=mu
			kdata[:,kc,:,:,kt]/=std
			
			torch.manual_seed(0)
			noise = sigma*torch.randn(kdata[:,kc,:,:,kt].shape).to(device)
			kdata[:,kc,:,:,kt]+= noise
			
			kdata[:,kc,:,:,kt] = std*kdata[:,kc,:,:,kt] + mu

	return kdata

class HOperator(nn.Module):

	"""
	The Operator H = F^H \circ F + lambda*Id

	"""

	def __init__(self, A, lambda_reg):
		super(HOperator, self).__init__()
		
		self.A = A
		self.lambda_reg = lambda_reg

	def forward(self, x):
		
		return self.A(x) + self.lambda_reg*x
	
class ConjGrad(nn.Module):

	"""
	The conjugate gradient block for solving the a linear problem
	Hx=b.

	"""

	def __init__(self):
		super(ConjGrad, self).__init__()
		
		
	def forward(self, H, x, b, niter=4):
		
		#x is the starting value, b the rhs;
		r = H(x)
		r = b-r
		
		#initialize p
		p = r.clone()
		
		#old squared norm of residual
		sqnorm_r_old = torch.dot(r.flatten(),r.flatten())
		
		for kiter in range(niter):
		
			#calculate Hp;
			d = H(p);
	
			#calculate step size alpha;
			inner_p_d = torch.dot(p.flatten(),d.flatten())
			alpha = sqnorm_r_old / inner_p_d
	
			#perform step and calculate new residual;
			x = torch.add(x,p,alpha= alpha.item())
			r = torch.add(r,d,alpha= -alpha.item())
			
			#new residual norm
			sqnorm_r_new = torch.dot(r.flatten(),r.flatten())
			print('||res_||_2^2 = {}'.format(sqnorm_r_new))
			
			#calculate beta and update the norm;
			beta = sqnorm_r_new / sqnorm_r_old
			sqnorm_r_old = sqnorm_r_new
	
			p = torch.add(r,p,alpha=beta.item())
	
		return x