# Parametric-Fokker-Planck-Equation

These are the Python codes for implementing Algorithm 4.1 mentioned in the paper [[1]](#1).

To run the algorithm, please direclty run `Main_algorithm.py`. 
It will produce plots of samples, estimated densities at chosen time nodes, as well as inner loss curves, graphs of Kantorovich dual function  <img src="https://latex.codecogs.com/gif.latex?\psi" />. 
Coordinates of samples, plot of KL loss curve and the hyperparameters of the algorithm are also recorded.

You may directly modify the hyperparameters of the algorithm in `Main_algorithm.py` or set up your own potential function <img src="https://latex.codecogs.com/gif.latex?V" /> based on your own interests.








## References
<a id="1">[1]</a> 
W. Li, S. Liu, H. Zha and H. Zhou, Neural Parametric Fokker-Planck Equations, arXiv preprint arXiv 2002.11309, (2020)
