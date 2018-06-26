# activelearningTCs
Adaptive stimulus selection (aka "active learning") for 1D and 2D tuning curves in
closed-loop experiments in Matlab.

**Description:** Adaptively selects stimuli from a 1D or 2D grid that
 maximize some notion of utility (e.g., information gain or minimal posterior
 variance), for Poisson neurons with tuning curves
 modeled as:
 1. a nonlinearly transformed Gaussian process (*demo1* and *demo2*).
 2. a parametric function (to appear).

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

* Examine the demo scripts for annotated example analyses of simulated
datasets: 
	*  `demo1_gpTCinference_1D.m` - demo for learning non-parametric 1D tuning curve under transformed GP prior
	*  `demo2_gpTCinference_2D.m` - demo for learning non-parametric 2D  tuning curve under transformed GP prior

