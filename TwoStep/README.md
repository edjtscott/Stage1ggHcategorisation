# Two-step Stage 1 classification

There are currently four scripts here that actually do things, one that submits the jobs to the IC batch, plus various helper scripts.
The main four are explained below.
The submission script, `runJobs.py`, is used because some of this is quite memory-intensive and the batch has higher limits than just running locally.
It allows you to specify which script you want to run (currently in a very crude way, just by un-commenting the relevant block of inputs) and in some cases different setups for each one.
For example, you can specify different options to pass to XGBoost when running the training and it will run each separately.
None of the jobs take longer than say fifteen minutes to run to completion.

## Building the diphoton BDT
The script `diphotonCategorisation.py` build a classifier to discriminate between signal and background in an almost-identical way to the "diphoton BDT" in the latest paper.
It can either pick up the ROOT files from the directory you specify, or load a saved dataframe if the code has already been run at least once.
The output is a .model file which contains the resulting classifier.

## Defining the categories
Once you have a diphoton BDT built, you can use it to specify category boundaries for each Stage 1 process.
This script currently assumes you want two categories per process, and optimises them by searching randomly through possible cut points.
There is no output, other than text in a log file which tells you some information about the optimal categories found.

## The VBF categories
Don't need to worry about this yet.

## Comparing data to Monte Carlo simulation
Same for this.
