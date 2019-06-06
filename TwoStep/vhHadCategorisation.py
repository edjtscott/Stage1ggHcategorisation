#usual imports
import ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system

from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import fill_hist
import usefulStyle as useSty

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams',default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')
trainFrac = 0.7
validFrac = 0.1

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'vh':'VH.root'}
theProcs = procFileMap.keys()

#define the different sets of variables used; will want to revise this for VH
allVars    = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv',
              'weight', 'CMS_hgg_mass', 'HTXSstage0cat', 'HTXSstage1_1_cat', 'cosThetaStar']
vhHadVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      trainFile   = upr.open('%s/%s'%(trainDir,fn))
      trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc]
      trainFrames[proc] = trainTree.pandas.df(allVars)
      trainFrames[proc]['proc'] = proc
  print 'got trees'
  
  #create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  #then filter out the events into only those with the phase space we are interested in
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass>100.]
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for vhHad BDT in the AN
  trainTotal = trainTotal[trainTotal.leadmva>-0.9]
  trainTotal = trainTotal[trainTotal.subleadmva>-0.9]
  trainTotal = trainTotal[trainTotal.leadptom>0.333]
  trainTotal = trainTotal[trainTotal.subleadptom>0.25]
  trainTotal = trainTotal[trainTotal.stage1cat>-1.]
  print 'done basic preselection cuts'
  
  #add extra info to dataframe
  print 'about to add extra columns'
  trainTotal['truthDipho'] = trainTotal.apply(truthDipho,axis=1)
  trainTotal['vhHadWeight'] = trainTotal.apply(vhHadWeight,axis=1)
  trainTotal['altDiphoWeight'] = trainTotal.apply(altDiphoWeight, axis=1)
  print 'all columns added'

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/trainTotal.pkl'%frameDir)
  print 'frame saved as %s/trainTotal.pkl'%frameDir

#read in dataframe if above steps done before
else:
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
  print 'Successfully loaded the dataframe'

sigSumW = np.sum( trainTotal[trainTotal.stage1cat>0.01]['weight'].values )
bkgSumW = np.sum( trainTotal[trainTotal.stage1cat==0]['weight'].values )
print 'sigSumW %.6f'%sigSumW
print 'bkgSumW %.6f'%bkgSumW
print 'ratio %.6f'%(sigSumW/bkgSumW)
#exit('first just count the weights')

#define the indices shuffle (useful to keep this separate so it can be re-used)
theShape = trainTotal.shape[0]
vhHadShuffle = np.random.permutation(theShape)
vhHadTrainLimit = int(theShape*trainFrac)
vhHadValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for vhHad training
vhHadX  = trainTotal[vhHadVars].values
vhHadY  = trainTotal['truthDipho'].values
vhHadTW = trainTotal['vhHadWeight'].values
vhHadAW = trainTotal['altDiphoWeight'].values
vhHadFW = trainTotal['weight'].values
vhHadM  = trainTotal['CMS_hgg_mass'].values
del trainTotal

vhHadX  = vhHadX[vhHadShuffle]
vhHadY  = vhHadY[vhHadShuffle]
vhHadTW = vhHadTW[vhHadShuffle]
vhHadAW = vhHadAW[vhHadShuffle]
vhHadFW = vhHadFW[vhHadShuffle]
vhHadM  = vhHadM[vhHadShuffle]

vhHadTrainX,  vhHadValidX,  vhHadTestX  = np.split( vhHadX,  [vhHadTrainLimit,vhHadValidLimit] )
vhHadTrainY,  vhHadValidY,  vhHadTestY  = np.split( vhHadY,  [vhHadTrainLimit,vhHadValidLimit] )
vhHadTrainTW, vhHadValidTW, vhHadTestTW = np.split( vhHadTW, [vhHadTrainLimit,vhHadValidLimit] )
vhHadTrainAW, vhHadValidAW, vhHadTestAW = np.split( vhHadAW, [vhHadTrainLimit,vhHadValidLimit] )
vhHadTrainFW, vhHadValidFW, vhHadTestFW = np.split( vhHadFW, [vhHadTrainLimit,vhHadValidLimit] )
vhHadTrainM,  vhHadValidM,  vhHadTestM  = np.split( vhHadM,  [vhHadTrainLimit,vhHadValidLimit] )

#build the background discrimination BDT
trainingDipho = xg.DMatrix(vhHadTrainX, label=vhHadTrainY, weight=vhHadTrainTW, feature_names=vhHadVars)
testingDipho  = xg.DMatrix(vhHadTestX,  label=vhHadTestY,  weight=vhHadTestFW,  feature_names=vhHadVars)
trainParams = {}
trainParams['objective'] = 'binary:logistic'
trainParams['nthread'] = 1
paramExt = ''
if opts.trainParams:
  paramExt = '__'
  for pair in opts.trainParams:
    key  = pair.split(':')[0]
    data = pair.split(':')[1]
    trainParams[key] = data
    paramExt += '%s_%s__'%(key,data)
  paramExt = paramExt[:-2]
print 'about to train vhHad BDT'
vhHadModel = xg.train(trainParams, trainingDipho)
print 'done'

#save it
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
vhHadModel.save_model('%s/vhHadModel%s.model'%(modelDir,paramExt))
print 'saved as %s/vhHadModel%s.model'%(modelDir,paramExt)

#build same thing but with equalised weights
altTrainingDipho = xg.DMatrix(vhHadTrainX, label=vhHadTrainY, weight=vhHadTrainAW, feature_names=vhHadVars)
print 'about to train alternative vhHad BDT'
altDiphoModel = xg.train(trainParams, altTrainingDipho)
print 'done'

#save it
altDiphoModel.save_model('%s/altDiphoModel%s.model'%(modelDir,paramExt))
print 'saved as %s/altDiphoModel%s.model'%(modelDir,paramExt)

#check performance of each training
vhHadPredYxcheck = vhHadModel.predict(trainingDipho)
vhHadPredY = vhHadModel.predict(testingDipho)
print 'Default training performance:'
print 'area under roc curve for training set = %1.3f'%( roc_auc_score(vhHadTrainY, vhHadPredYxcheck, sample_weight=vhHadTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( roc_auc_score(vhHadTestY, vhHadPredY, sample_weight=vhHadTestFW) )

altDiphoPredYxcheck = altDiphoModel.predict(trainingDipho)
altDiphoPredY = altDiphoModel.predict(testingDipho)
print 'Alternative training performance:'
print 'area under roc curve for training set = %1.3f'%( roc_auc_score(vhHadTrainY, altDiphoPredYxcheck, sample_weight=vhHadTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( roc_auc_score(vhHadTestY, altDiphoPredY, sample_weight=vhHadTestFW) )
