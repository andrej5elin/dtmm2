# dtmm2
 
DTMM2 is a temporary code from https://zenodo.org/badge/latestdoi/125330690, hosted here: https://github.com/IJSComplexMatter/dtmm
The code was modified to include second harmonic generation calculation on 1D structures using Berreman 4x4 matrix formalism. Currently it only works in 1D - no diffraction.
The code was also modified to support the Jones vector representation of vectorial beams as explained in "Focused beam propagation in cholesteric liquid crytals", which is accepted for publication in Optics Express.

This is an ongoing project. The goal is to develop a new interface for numerical simulations on birefringent material for POM and SHG imaging. We plan to backport the code to the original *dtmm* package once the work is complete.

To install do

$ python setup.py install

In the examples folder, you will find the code which generates images from the paper https://arxiv.org/pdf/2404.18619 and for paper "Focused beam propagation in cholesteric liquid crytals", which is accepted for publication in Optics Express.
