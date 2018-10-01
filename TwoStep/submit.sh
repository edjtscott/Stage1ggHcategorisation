#!/bin/bash
# this script submits whatever script(s) you're using to the batch
# typical usage (high memory requirements):
# qsub -q hep.q -o $PWD/submit.log -e $PWD/submit.err -l h_vmem=24G submit.sh

#inputs
SCRIPT=diphotonCategorisation.py
MYDIR=/vols/build/cms/es811/FreshStart/STXSstage1/Classification/Pass0/CMSSW_10_2_0/src/Stage1categorisation/TwoStep
RAND=$RANDOM

#execution
cd $MYDIR
eval `scramv1 runtime -sh`
cd $TMPDIR
mkdir -p scratch_$RAND
cd scratch_$RAND
cp -p $MYDIR/*.py .
echo "About to run the following command:"
echo "python $SCRIPT"
if ( python $SCRIPT ) then
  touch $MYDIR/submit.done
  echo 'Success!'
else
  touch $MYDIR/submit.fail
  echo 'Failure..'
fi
cd -
rm -r scratch_$RAND
