# MOPSI-Hypocoercivity
Project aiming at implementing Finite Difference Schemes to approximate Fokker-Planck Equations' solutions. 

Goal is to showcase exponential error decrement in cases with no coercivity guarantee using techniques described in a Sept. 2015 paper by F. Achleitner, A. Arnold, D. Stürzer.

Julia file provides a n-dimensional PDE solver via finite differences. Notebook provides plots reconstituting 2015 paper accurately.

NB: PDEs above dimenson 3 are unlovable using this method in practice.
