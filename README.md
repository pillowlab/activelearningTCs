# activelearningTCs
Adaptive stimulus selection (aka "active learning") for 1D and 2D tuning curves for
closed-loop experiments in Matlab.

**Description:** Selects optimal stimuli from a 1D or 2D grid
 according to infomax (maximizing mutual information between response
 and model parameters) or uncertainty sampling (stimulus for which
 tuning curve has maximal uncertainty), for Poisson neurons with
 tuning curves modeled as either:
 1. a parametric function (*demo1*). 
 2. a nonlinearly transformed Gaussian process (*demo2* and *demo3*).

**Relevant publications:**

*  Pillow & Park (2016). **Adaptive Bayesian methods for closed-loop
   neurophysiology**. [[link]](http://pillowlab.princeton.edu/pubs/abs_Pillow16_ActiveLearningChap.html)

*  Park, Weller, Horwitz, & Pillow (2014).  **Bayesian active learning
   of neural firing rate maps with transformed Gaussian process
   priors. Neural Computation**, *Neural Computation* 2014 [[link]](http://pillowlab.princeton.edu/pubs/abs_ParkM_GPactivelearning_NC14.html)

Download
==========

* **Download**:   zipped archive  [activelearningTCs-master.zip](https://github.com/pillowlab/activelearningTCs/archive/master.zip)
* **Clone**: clone the repository from github: ```git clone https://github.com/pillowlab/activelearningTCs.git```

Usage
=====

* Launch matlab and cd into the directory containing the code
 (e.g. `cd code/activelearningTCs/`).

* Examine the demo scripts for annotated example analyses of simulated datasets:
	*  `demo1_parametricTC_1Dinfomax` - demo for parametric 1D tuning curve with infomax stim selection
	*  `demo2_gpTC_1D.m` - demo for non-parametric 1D tuning curve under transformed GP prior
	*  `demo3_gpTC_2D.m` - demo for  non-parametric 2D  tuning curve under transformed GP prior

