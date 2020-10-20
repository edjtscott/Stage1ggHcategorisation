# VBF STXS stage 1.2 classification

Here we train the so-called dijet BDT to output three scores per event, corresponding to how similar the event is to typical VBF, ggH, or background events.
The name of the script is called vbfTraining.py; it does quite a few things, but most of it is simply reading in the data and selecting appropriate events.
More details are given below. 

The submission script, `runJobs.py`, is used because some of this is quite memory-intensive and the batch has higher limits than just running locally.
It allows you to specify which script you want to run, and whether you want to run it locally or on the IC batch system. 
None of the jobs should take longer than say fifteen minutes to run to completion.

## The VBF training
The basic details of the script are as follows:
- read in the data for each physics process (e.g. VBF, ggH, non-Higgs backgrounds)
- separate the data into training and test sets. The first is used to train the BDT, the second is used to evaluate its performance (without bias).
- set up the BDT and train it
- evaluate its performance using the area under the ROC curve 

Some possible nice additions to this scipt would be:
- adding plots of the different variables, so that one can compare how they look e.g. between signal VBF events and the other classes (ggH and background)
- modifying which input variables are used, to see how the performance changes
- modifying the hyperparameters of the BDT, to see how the performance changes
- performing repeated runs with different random seeds, to establish how much variance there is due to different sets of event being used

## Defining analysis categories
We will add this later!

