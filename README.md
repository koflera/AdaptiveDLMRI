# AdaptiveDLMRI

Here, we provide a minimal example code to our paper 

"Adaptive Sparsity Level and Dictionary Size Estimation for Image Reconstruction in Accelerated 2D Radial Cine MRI"

by M.C. Pali, T. Schaeffter, C. Kolbitsch and A. Kofler.

With the code you should be able to run the reconstruction algorithm for three different combinations of dictionary learning (DL) and sparse coding (SC)  algorithms, i.e. K-SVD + OMP, ITKrM + OMP and aITKrM + aOMP as shown in the paper.

As a toy example, you can apply DL and SC for denosing an image. For this, just run the script basic_example.py.
The image was borrowed from http://jtl.lassonde.yorku.ca/software/datasets/.

For all theoretical aspects of aTIKrM and aOMP, please visit Karin Schnass' website: https://www.uibk.ac.at/mathematik/personal/schnass/.

The code was written by M. C. Pali and A. Kofler at the University of Innsbruck (Austria), the Physikalisch-Technische Bundesanstalt, Berlin and Braunschweig and at the Charité - Universitätsmedizin Berlin.

If you use the code or find it useful, please cite our work:

@article{pali2020adaptive,
  title={Adaptive Sparsity Level and Dictionary Size Estimation for Image Reconstruction in Accelerated 2D Radial Cine MRI},
  author={Pali, Marie-Christine and Schaeffter, Tobias and Kolbitsch, Christoph and Kofler, Andreas},
  journal={Medical Physics},
  year={2020},
  DOI={10.1002/mp.14547},
  publisher={Wiley Online Library}
}

## For running the code, you will need the following main packages:
- PyTorch: 1.6.0.
- TorchKbNufft: 0.3.4.
- Numpy: 1.19.2.
- SciKit-Learn: 0.23.2.
- ksvd: 0.0.3.
