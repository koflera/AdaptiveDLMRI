"""
Created on Wed Oct 14 

@author: M.C.Pali & A. Kofler

"""
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
from sklearn.feature_extraction import image

import os
from termcolor import colored
from sklearn.decomposition import DictionaryLearning 
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn import preprocessing
from ksvd import ApproximateKSVD



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# ITKrM algorithm (fast with matrices)
def itkrm(data,K,S,maxit,dinit):
	""" Iterative Thresholding and K-residual Means (ITKrM) algorithm
	:param data: training signals
	:param K: number of dictionary atoms
	:param S: sparsity level
	:param maxit: maximal number of dictionary learning iterations
	:param dinit: initial dictionary
	:returns: learned dictionary 
	"""
	
	""" preprocessing """
	dold = np.asmatrix(dinit)
	data = np.asmatrix(data)
	
	d,N = np.shape(data)
	
	
	""" algorithm """
	for it in range(maxit):
		''' thresholding '''
		ip = np.dot(dold.transpose(),data)
		
		absip = np.abs(ip)
		signip = np.sign(ip)
		I = np.argsort(absip, axis=0)[::-1]
		
		gram = np.dot(dold.transpose(),dold)
		dnew = np.asmatrix(np.zeros((d,K)))
		X = np.zeros((K,N))
		
		''' update coefficient matrix '''
		It = I[0:S,:]  # indices of S largest inner products
		for n in range(N):
			In = It[:,n]
			try:
				coeff = np.linalg.solve(gram[In,np.transpose(np.asmatrix(In))],np.asmatrix(ip[In,n]))
			except:
				pinv_gram_In = np.linalg.pinv(gram[In,np.transpose(np.asmatrix(In))])
				coeff = np.dot(pinv_gram_In,np.asmatrix(ip[In,n]))
			X[In,n] = coeff
		app = np.dot(dold,X)
		avenapp = np.linalg.norm(app,'fro')
		res = data - app
		
		''' dictionary update '''
		for j in range(K):
			# signals that use atom j
			J = np.ravel(X[j,:].ravel().nonzero())
			dnew[:,j] = np.dot(res[:,J],signip[j,J].transpose())
			dnew[:,j] = dnew[:,j] + np.dot(dold[:,j],np.sum(absip[j,J]))
		
		# do not update unused atoms
		avenapp = avenapp/N
		scale = np.sum(np.multiply(dnew,dnew),axis=0)
		nonzero = np.argwhere(scale > (0.001*avenapp/d))[:,1]
		
		dnew[:,nonzero] = np.dot(dnew[:,nonzero],np.diagflat(np.reciprocal(np.sqrt(scale[:,nonzero]))))
		dold[:,nonzero] = dnew[:,nonzero]
		
	dico = dold
	return dico

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# adaptive OMP
def a_omp_plus1(dico,data):
	""" Adaptive Orthogonal Matching Pursuit (aOMP) algorithm for sparse coding
	:param dico: dictionary used for sparse coding
	:param data: signals to be sparsely approximated, stored in a matrix
	:returns: sparse coefficient matrix 
	"""
	
	""" preprocessing """
	dico = np.asmatrix(dico)
	data = np.asmatrix(data)
	
	d,N = np.shape(data)
	d,K = np.shape(dico)
	
	Smax = np.int(np.floor(d/np.log(K)))                   # maximal number of coefficients for each signal
	threshold1 = np.sqrt(2*(np.log(2*K) - np.log(0.25))/d) # threshold for 1st iteration (tau_1)
	threshold = np.sqrt(2*(np.log(2*K) - np.log(0.5))/d)   # threshold for aOMP in subsequent iterations (tau_2)
	
	# initialisation
	X = np.zeros((K,N))
	iterations_ps = np.zeros((N,1)) # iterations for each signal
	
	""" algorithm """
	for n in range(N):
		yn = data[:,n]                   # signal y_n
		norm_yn = np.linalg.norm(yn)
		res = yn                         # initial residual
		
		ind_yn = np.zeros((Smax,1))
		
		it = 0
		
		''' part of aOMP using threshold tau_1 '''
		resip = np.dot(dico.transpose(),res) # residual inner product
		abs_resip = np.abs(resip) # absolute value of residual inner product
		
		# find first part of support
		supp_new = np.argwhere(abs_resip > threshold1*np.linalg.norm(res))
		supp_new = np.array(supp_new[:Smax,0])
		
		l = np.int(supp_new.size)		
		it = it + l
		ind_yn[0:l,:] = np.asarray([supp_new]).transpose()
		ind_it = np.ravel(np.asmatrix(ind_yn[0:l,:],dtype=int))
		
		# compute coefficient vector and new residual
		if len(supp_new)!=0:
			try:
				x = np.linalg.solve(dico[:,supp_new],yn)
			except:
				x = np.dot(np.linalg.pinv(dico[:,supp_new]),yn)
			
			res = yn - np.dot(dico[:,supp_new],x)
		else:
			res = yn
		
		
		''' part of aOMP using threshold tau_2 '''
		abs_resip = np.abs(np.dot(dico.transpose(),res)) # absolute value of residual		
		max_val = np.max(abs_resip)              # maximal absolute value of inner product between dictionary and residual
		max_pos = np.int(np.argmax(abs_resip))   # position of max_val
		
		while max_val >= threshold*np.linalg.norm(res) and np.linalg.norm(res) > (10**-3)*norm_yn and it < Smax:
			ind_yn[it,:] = max_pos
			ind_it = np.ravel(np.asmatrix(ind_yn[0:(it+1),:],dtype=int)) # current support
			
			# compute coefficient vector and new residual
			if len(ind_it)!=0:
				try:
					x = np.linalg.solve(dico[:,ind_it],yn)
				except:
					x = np.dot(np.linalg.pinv(dico[:,ind_it]),yn)
				res = yn - np.dot(dico[:,ind_it],x)

			else:
				res = yn
			
			it = it+1
			
			# new residual inner product
			resip = np.dot(dico.transpose(),res)
			abs_resip = np.abs(resip)
			max_val = np.max(abs_resip)
			max_pos = np.int(np.argmax(abs_resip))
			
		if it > 0:

			if len(ind_it)!=0:
				# store sparse coefficient vector in sparse coefficient matrix X
				X[np.array(ind_it),n] = np.ravel(x)

			iterations_ps[n,:] = it
	
	return X


#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------
"""
-- dictionary learning via ITKrM and adaptively chosen sparsity level S and dictionary size K
"""
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------
# sub-functions for adaptive ITKrM:
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------

# one iteration of adaptive ITKrM (fast with matrices)
def batch_a_itkrm1(dico,data,S,repinit,minobs):
	"""
	Subfunction of aITKrM:
	-- one iteration of adaptive Iterative Thresholding and K-residual Means (aITKrM) algorithm --
	:param dico: current dictionary estimate
	:param data: training signals
	:param S: sparsity level used for thresholding
	:param repinit: initialisation for replacement candidates
	:param minobs: minimal number of required observations
	
	:returns:
		dico: learned dictionary
		freq: score for each atom in the dictionary
		repcand: candidate atoms
		repfreq: score of candidate atoms
		Sguess: estimated sparsity level
		avenres: average residual energy
	"""
	
	""" preprocessing """
	dico = np.asmatrix(dico)
	data = np.asmatrix(data)
	
	d,N = np.shape(data)
	d,K = np.shape(dico)
	
	S = np.int(S)
	
	# initialisation
	L = np.int(np.round(np.log(d))) # number of learned replacement candidates
	m = np.int(np.round(np.log(d))) # number of iterations for learning candidates
	NL = np.int(np.floor(N/m))      # number of training signals for candidates
	
	candtau = 2*np.log(2*NL/d)/d # threshold for candidates
	tau = 2*np.log(2*N/minobs)/d # threshold dictionary atoms
	
	
	""" algorithm """
	
	# initialisation
	dnew = np.asmatrix(np.zeros((d,K))) # new dictionary
	freq = np.zeros((1,K)) # score for each atom in the dictionary
	X = np.zeros((K,N))    # coefficient matrix
	Z = np.zeros((K,N))    # residual inner products + coefficients
	
	repcand = repinit
	repcandnew = np.asmatrix(np.zeros((d,L))) # new replacement candidate
	repfreq = np.zeros((1,L))
	
	avenres = 0  # average energy of residual
	avenapp = 0  # average energy of approximation
	Sguess = 0   # guess for sparsity level
	
	
	''' thresholding '''
	ip = np.dot(dico.transpose(),data)
	absip = np.abs(ip)
	signip = np.sign(ip)
	I = np.argsort(absip, axis=0)[::-1]
	It = I[0:S,:]  # indices of S largest inner products in absolute value
	
	gram = np.dot(dico.transpose(),dico)	
	
		
	''' update coefficient matrix '''
	for n in range(N):
		In = It[:,n]
		try:
			coeff = np.linalg.solve(gram[In,np.transpose(np.asmatrix(In))],np.asmatrix(ip[In,n]))
		except:
			pinv_gram_In = np.linalg.pinv(gram[In,np.transpose(np.asmatrix(In))])
			coeff = np.dot(pinv_gram_In,np.asmatrix(ip[In,n]))
		X[In,n] = coeff
	
	app = np.dot(dico,X)
	res = data - app
	
	''' dictionary update '''
	for j in range(K):
		# signals that use atom j
		J = np.ravel(X[j,:].ravel().nonzero())
		dnew[:,j] = np.dot(res[:,J],signip[j,J].transpose())
		dnew[:,j] = dnew[:,j] + np.dot(dico[:,j],np.sum(absip[j,J]))
	
	
	""" part for adaptivity """
	
	# counter for adaptivity
	res_energy = (np.linalg.norm(res))**2  # energy of residual
	app_energy = (np.linalg.norm(app))**2  # energy of signal approximation
	
	enres = np.sum(np.square(res),axis=0)
	enapp = np.sum(np.square(app),axis=0)
	avenres = np.linalg.norm(res,'fro')/N
	avenapp = np.linalg.norm(app,'fro')/N
	
	# steps for estimating the sparsity level
	sig_threshold = np.sqrt(app_energy/d+res_energy*2*np.log(4*K)/d)
	Z = X + np.dot(dico.transpose(),res)  # inner products with residual + coefficients
	Sguess = np.sum(np.ceil(np.sum(np.abs(Z),axis=1) - sig_threshold*np.ones((K,1))))/N
	
	# counting above threshold
	threshold = enres*tau + enapp/d
	freq = np.sum(np.square(np.abs(X)) > threshold,axis=1)  # atom scores
	
	
	''' candidate update '''    
	for it in range(m):
		res_cand = res[:,it*NL:(it+1)*NL]
		enres = np.sum(np.square(res_cand),axis=0)
		repip = np.dot(np.transpose(repcand),res_cand)
		signip = np.sign(repip)
		max_repip_val = np.max(np.abs(repip),axis=0)
		max_repip_pos = np.argmax(np.abs(repip),axis=0)
		
		for j in range(L):
			# signals that use atom j
			J = np.argwhere(max_repip_pos == j)[:,1]
			repcandnew[:,j] = res_cand[:,J]*np.transpose(np.sign(repip[j,J]))
			
			if it == m-1:
				# candidate counter
				repfreq[:,j] = np.sum(np.square(max_repip_val[:,J]) >= enres[:,J]*candtau)
		
		repcand = preprocessing.normalize(repcandnew,norm='l2',axis=0) # normalise candidate atoms
	
	
	''' dictionary update - remove unused atoms '''
	scale = np.sum(np.square(dnew),axis=0)
	nonzero = np.argwhere(scale > (0.001*avenapp/d))[:,1]
	iszero = np.argwhere(scale <= (0.001*avenapp/d))[:,1]
	
	# remove unused atoms
	freq[np.ravel(iszero)] = 0
	dnew[:,np.ravel(nonzero)] = np.dot(dnew[:,np.ravel(nonzero)],np.diagflat(np.reciprocal(np.sqrt(scale[:,np.ravel(nonzero)]))))
	dico[:,np.ravel(nonzero)] = dnew[:,np.ravel(nonzero)]
	
	# normalise atoms
	dico = preprocessing.normalize(dico,norm='l2',axis=0)
	repcand = preprocessing.normalize(repcandnew,norm='l2',axis=0)
	
	repfreq = repfreq/d
	freq = freq/minobs
		
	return dico, freq, repcand, repfreq, Sguess, avenres



def prune_coherent(dico,freq,mumax):
	""" Subfunction of aITKrM:
	-- prune coherent atoms in a dictionary --
	:param dico: dictionary matrix with coherent atoms to replace
	:param freq: usage scores for each dictionary atom
	:param mumax: maximal allowed coherence
	
	:returns:
		dico: pruned dictionary
		freq: usage scores of atoms in pruned dictionary
		npruned: number of pruned atoms
	"""
	
	""" preprocessing """
	dico = np.asmatrix(dico)
	freq = np.asmatrix(freq)
	
	d,K = np.shape(dico)
	m,Kf = np.shape(freq)
	
	
	""" pruning coherent atoms """
	
	prune = np.asarray([],dtype=int)
	gram = np.dot(dico.transpose(),dico)
	holgram = np.abs(gram-np.identity(K))     # hollow gram-matrix
	maxcorr = np.max(np.max(holgram,axis=0))  # maximal correlation
	
	while maxcorr > mumax:
		x = np.argwhere(holgram == maxcorr)
		rowi = x[0,0]
		coli = x[0,1]
		
		hh = np.sign(np.dot(dico[:,rowi].transpose(),dico[:,coli]))
		newatom = np.add(dico[:,rowi]*freq[0,rowi],dico[:,coli]*freq[0,coli]*hh)
		
		dico[:,rowi] = newatom
		if np.linalg.norm(newatom) > 0:
			dico[:,rowi] = newatom/np.linalg.norm(newatom)
		
		freq[0,rowi] = freq[0,rowi] + freq[0,coli]
		prune = np.append(prune,np.asarray([coli],dtype=int))
		holgram[:,[rowi,coli]] = 0
		holgram[[rowi,coli],:] = 0
		maxcorr = np.max(np.max(holgram,axis=0))
	
	dico = np.delete(dico,prune,axis=1)
	freq = np.delete(freq,prune,axis=1)
	npruned = np.size(prune)
	
	return dico, freq, npruned


"""
-- prune unused atoms in a dictionary --
"""

def prune_unused(dico,freq,threshold):
	""" Subfunction of aITKrM:
	-- prune unused atoms in a dictionary --
	:param dico: dictionary matrix with unused atoms to prune
	:param freq: matrix counting how often/strongly atoms have been used in the last m iterations
	:param threshold: cutoff threshold
	
	:returns:
		dico: pruned dictionary
		freq: usage scores of atoms in pruned dictionary
		npruned: number of atoms that have been pruned
	"""
	
	
	""" preprocessing """
	dico = np.asmatrix(dico)
	freq = np.asmatrix(freq)
	
	d,K = np.shape(dico)
	m,Kf = np.shape(freq)	
	
	maxprune = np.int(np.minimum(np.round(d/5),np.floor(K/2))) # maximal number of pruned atoms
	
	
	""" pruning unused atoms """
	
	# compare maximal score in the last m runs to threshold
	if m > 1:
		score = np.max(freq,axis=0)
	else:
		score = freq
	
	notused = np.argwhere(score < threshold) # unused atoms - atoms with score below threshold
	notused = notused[:,1]
	
	# if more than max number of atoms are below --> delete those with smallest score
	if np.size(notused) > maxprune:
		I = np.argsort(score)
		notused = I[:,0:maxprune]
	
	dico = np.delete(dico,notused,axis=1) # prune unused atoms
	freq = np.delete(freq,notused,axis=1)
	npruned = np.size(notused)
	
	return dico, freq, npruned



def add_atoms(dico,freq,repcand,repfreq,mumax,M):
	""" Subfunction of aITKrM:
	-- add promising candidates to a dictionary --
	:param dico: dictionary matrix
	:param freq: usage scores of dictionary atoms
	:param repcand: candidate atoms to add
	:param repfreq: usage scores for replacement candidates
	:param mumax: maximal allowed coherence between atoms
	:param M: value for newly added atoms
	
	:returns:
		dico: dictionary with added atoms
		freq: usage scores of old and new atoms
		nadded: number of added atoms
	"""
	
	""" preprocessing """
	dico = np.asmatrix(dico)
	repcand = np.asmatrix(repcand)
	freq = np.asmatrix(freq)
	repfreq = np.asmatrix(repfreq)
	
	d,K = np.shape(dico)
	dl,L = np.shape(repcand)
	m,Kf = np.shape(freq)
	mr,Lf = np.shape(repfreq)
	
	nadded = 0
	
	
	""" algorithm """
	
	repfreq = np.sort(repfreq)[::-1]
	repind = np.argsort(repfreq)[::-1]
	repcand = repcand[:,np.ravel(repind)]
	num_promise = np.argwhere(repfreq > 1)[:,1].size # number of promising candidates
	
	for ell in range(num_promise):
		# maximal coherence between dictionary and each promising candidate
		rcoh = np.max(np.abs(np.dot(dico.transpose(),repcand[:,ell])))
		# check coherence
		if rcoh < mumax:
			dico = np.concatenate((dico,repcand[:,ell]),axis=1)
			freq = np.concatenate((freq, M*np.ones((m,1))),axis=1)
			nadded = nadded + 1
	
	return dico, freq, nadded

#------------------------------------------------------------------------------
# end of sub-functions
#------------------------------------------------------------------------------

"""
-- dictionary learning with adaptively chosen sparsity level and dictionary size -- 
"""

# dictionary learning via ITKrM and adaptively chosen sparsity level 
def a_itkrm(data,K,maxit,dinit):  
	""" Adaptive Iterative Thresholding and K-residual Means (aITKrM) algorithm
	:param data: training signals
	:param K: initial number of dictionary atoms
	:param maxit: maximal number of dictionary learning iterations
	:param dinit: initial dictionary
	
	:returns:
		dico: learned dictionary
		S: new sparsity level
	"""
		
	
	""" preparations/preprocessing """
	
	data = np.asmatrix(data)
	d,N = np.shape(data)
	
	# initialisation/parameters
	Ke = d          # initial estimate of dictionary size (number of atoms)
	Se = np.int(1)  # initial estimate of sparsity level
	
	mumax = 0.7    # maximal allowed coherence
	
	minobs = np.round(d) # minimal number of required observations
	
	m = np.int(np.round(np.log(d)))       # protected runs for newly added atoms
	L = np.int(np.round(np.log(d)))       # number of candidates learned per iteration
	
	Smax = np.int(np.round(d/np.log(d)))  # maximal allowed sparsity level
	Kmax = np.int(np.round(d*np.log(d)))  # maximal allowed number of dictionary atoms
	
	Ke = np.int(np.min([Kmax,Ke]))
	Se = np.int(np.min([Se,Smax,Ke])) 
	
	# initialisation of dictionary and replacement candidates	
	dinit = np.random.randn(d,Ke) # random initialisation
	dinit = preprocessing.normalize(dinit,norm='l2',axis=0)
	repcandinit = np.random.randn(d,maxit*L)
	
	# initialisations for statistics to track
	npruned_coh = np.zeros((1,maxit)) # number of pruned coherent atoms
	npruned_uu = np.zeros((1,maxit))  # number of pruned unused atoms
	nadded = np.zeros((1,maxit))      # number of added atoms
	
	K_history = Ke*np.ones((1,maxit))
	S_history = Se*np.ones((1,maxit))
	
	# initialisation of dictionary, size, sparsity level and usage score
	rdico = dinit
	Kr = Ke
	Sr = Se
	rfreqm = np.ones((m,Kr))    
	
	""" algorithm """
	
	for it in range(maxit):
		# initialisation of replacement candidates
		repcand = repcandinit[:,(it*L):((it+1)*L)]
		repcand = preprocessing.normalize(repcand,norm='l2',axis=0)
		
		''' dictionary learning '''
		# one iteration of adapted ITKrM (sub-function itkrm1)
		(rdico,rfreq,repcand,repfreq,Sguess,avenres) = batch_a_itkrm1(rdico,data,Sr,repcand,minobs)
		
		
		# update atom scores
		rfreqm[1:m,:] = rfreqm[0:m-1,:]
		rfreqm[0,:] = np.ravel(rfreq)
		
		''' prune coherent and unused atoms '''
		# prune coherent atoms
		if it >= m and Sr > 0:
			(rdico,rfreqm,npruned) = prune_coherent(rdico,rfreqm,mumax)
			npruned_coh[0,it] = npruned
		
		# prune unused atoms
		if it >= 2*m-1:
			maxscore = np.max(rfreqm[0,:]) # best score in last iteration
			if maxscore > 1:
				(rdico,rfreqm,npruned) = prune_unused(rdico,rfreqm,np.minimum(1,maxscore/2))
				npruned_uu[0,it] = npruned
			else:  # safeguard for ill-chosen number of observations and data that is not sparse
				(rdico,freqm,npruned) = prune_unused(rdico,rfreqm,maxscore/np.sqrt(d))
				npruned_uu[0,it] = npruned
		
		
		''' add promising candidates '''
		if it >= m-1 and it < maxit-3*m-1:
			(rdico,rfreqm,nadd) = add_atoms(rdico,rfreqm,repcand,repfreq,mumax,1)
			nadded[0,it] = nadd
		
		Kr = rdico.shape[1]
		K_history[0,it] = Kr
		
		if Kr > Kmax:
			rdico = rdico[:,0:Kmax]
			Kr = Kmax
		
		''' adapt sparsity level '''
		if it > m-1:
			Sr = Sr + np.sign(np.round(Sguess)-Sr)
			Sr = np.minimum(Smax,Sr)
			Sr = np.maximum(Sr,1)
			
			S_history[0,it] = Sr
		else:
			Sr = np.minimum(Sr,Kr)
		
	return rdico, S_history

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

"""
-- reconstruct matrix from 3D patches --
"""

def sparse_approx(data,dico,sparse_coding_algo,S): 
	# Input: data - a matrix of shape (N,d) of N d-dim signals to be sparse-coded
	# 	 	 dico - the dictionary to be used for te sparse approximation
	# 	 	 sparse_coding_algo - the algorithm for sparse coding;
	# 	 	 S - the number of atoms used in the sparse approximation;
	
	# Output: all signals sparsely approximated ;
	
	# sparse coding step
	t_sc = time.time()  

	

	if sparse_coding_algo == 'omp':

		t0 = time.time()
		coder = OrthogonalMatchingPursuit(n_nonzero_coefs=S)
		code = coder.fit(dico,data).coef_  # has shape NxK
		t1 = time.time()
		print(colored('time needed for omp: {}'.format( t1-t0),'green'))
		data = np.dot(dico,code.transpose()).transpose() 

	else:
		
		if sparse_coding_algo=='a_omp_plus1':
			t0 = time.time()
			code = a_omp_plus1(dico,data)
			t1 = time.time()
			print(colored('time needed for a_omp_plus1: {}'.format( t1-t0),'green'))
			
		elif sparse_coding_algo=='a_omp':
			t0 = time.time()
			code = a_omp(dico,data)
			t1  = time.time()
			print(colored('time needed for a_omp: {}'.format( t1-t0),'green'))
			
		elif sparse_coding_algo=='APm':
			t0 = time.time()
			code = APm(data,dico)
			t1  = time.time()
			print(colored('time needed for APm: {}'.format( t1-t0),'green'))
			
		#sarse approx as matrix-vector multiplication;
		data = np.dot(dico,code).transpose() 
	
	t_sc = time.time() - t_sc  
	print(colored('time used for sparse coding: ' + str(t_sc) + ' sec','green'))
	
	return data, t_sc, 


"""
-- learn dictionary from data --
"""
def learn_dico_from_data(data,
						 dico_learn_algo,
						 dico0,                   
						 S=4,                   
						 K=128,           
						 n_iter_DL=10):        
	
	# Input:
	# data - signals from which we want to learn the dictionary (shape = (N,d)), N d-dimensional signals
	#        ...
	# Output: dictionary, sparsity level, time for dictionarby learning

	""" start dictionary learning """
	t_DL = time.time()  

	if dico_learn_algo == 'itkrm':   
		
		t0 = time.time()
		print(colored('dictionary learning via ITKrM','green'))
		dico = itkrm(data,K,S,n_iter_DL,dico0)
		t1 = time.time() 
		print(colored('time needed {} itrkm iterations: {}'.format(n_iter_DL, t1-t0),'green'))
	
	elif dico_learn_algo == 'a_itkrm':
		print(colored('dictionary learning via adaptive ITKrM','green'))
		t0 = time.time()
		(dico,S_history) = a_itkrm(data,K,n_iter_DL,dico0)
		t1 = time.time() 
		print(colored('time needed {} a_itrkm iterations: {}'.format(n_iter_DL, t1-t0),'green'))
		
		S = np.int(S_history[0,n_iter_DL-1])

	elif dico_learn_algo =='ksvd':
		print(colored('dictionary learning via KSVD','green'))
		#define the model;
		aksvd = ApproximateKSVD(n_components=K,max_iter=n_iter_DL)
		t0 = time.time()
		dico = aksvd.fit(data.transpose()).components_.transpose()
		t1 = time.time()  
		print(colored('time needed {} ksvd iterations: {}'.format(n_iter_DL, t1-t0),'green'))

	t_DL = time.time() - t_DL
	
	return dico, S, t_DL


  