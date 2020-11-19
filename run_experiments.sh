#Examples on how to run the reconstruction code;
#N.B. for being able to use Psi_training=0, you have to have a pre-trained dictionary in one of the respective folders.

#experiment with K-SVD + OMP, K =128, S=8; dictionary learned during the reconstruction;
python run_dico_MRI_reco.py  --DL=ksvd --SC=omp --K=128 --S=8 --T=8 --Psi_training=1 

#experiment with ITKrM + OMP, K =64, S=4; dictionary learned during the reconstruction;
python run_dico_MRI_reco.py  --DL=itkrm --SC=omp --K=64 --S=4 --T=8 --Psi_training=1

#the combination aITKrM + aOMP; (sparsity level S and number of atoms K are estimated)
python run_dico_MRI_reco.py  --DL=a_itkrm --SC=a_omp_plus1 --T=8  --Psi_training=1



