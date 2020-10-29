# Stage1categorisation
Code to train various classifiers, aiming to optimise categorisation for the Higgs to diphoton analysis at Stage 1 of the Simplified Template Cross-section (STXS) framework.

## Setup
This is all desiged to run on the IC LX machines, in a CMSSW 10 environment.
Below are some instructions which should get you up and running.
Starting from a clean area:

```bash
cmsrel CMSSW_10_2_0
cd CMSSW_10_2_0/src/
cmsenv
git cms-init
git clone -b Clean2020 https://github.com/edjtscott/Stage1categorisation.git
cd Stage1categorisation
```

If you don't have a github account yet, it's probably worth creating one and forking this repository so you can play with it and update as needed.
Once you do, then you'd add something like

```bash
git remote add yourusername https://github.com/yourusername/Stage1categorisation.git
git checkout -b yourbranchname
```

to create a new branch.
As you can see, there is currently just one directory, `VBF`.
There the VBF part of the analysis is run; we will add more parts later if needed.
