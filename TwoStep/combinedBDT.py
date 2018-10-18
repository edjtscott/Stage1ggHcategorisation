#usual imports
import ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system

from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, combinedWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', help='Name of dataframe if it already exists')
parser.add_option('-m','--modelName', default=None, help='Name of model for testing')
parser.add_option('--trainParams',default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models').replace('ForVBF/','')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')
trainFrac = 0.7
validFrac = 0.1

#define the different sets of variables used
diphoVars    = ['leadmva','subleadmva','leadptom','subleadptom',
                'leadeta','subleadeta',
                'CosPhi','vtxprob','sigmarv','sigmawv']

#get trees from files, put them in data frames
#procFileMap = {'vbf':'VBF.root', 'dipho':'Dipho.root', 'gjet':'GJet.root', 'qcd':'QCD.root'}
procFileMap = {'vbf':'VBF.root', 'dipho':'Dipho.root', 'gjet':'GJet.root'}
theProcs = procFileMap.keys()

#either make or load dataframe
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      trainFile   = r.TFile('%s/%s'%(trainDir,fn))
      if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
      else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
      trainTree.SetBranchStatus('*',0)
      trainTree.SetBranchStatus('CMS_hgg_mass',1)
      trainTree.SetBranchStatus('weight',1)
      trainTree.SetBranchStatus('VBFMVAValue',1)
      trainTree.SetBranchStatus('diphomvaxgb',1)
      trainTree.SetBranchStatus('stage1cat',1)
      trainTree.SetBranchStatus('dijet_Mjj',1)
      trainTree.SetBranchStatus('dijet_LeadJPt',1)
      trainTree.SetBranchStatus('dijet_SubJPt',1)
      trainTree.SetBranchStatus('leadptom',1)
      trainTree.SetBranchStatus('subleadptom',1)
      trainTree.SetBranchStatus('leadmva',1)
      trainTree.SetBranchStatus('subleadmva',1)
      for var in diphoVars: trainTree.SetBranchStatus(var,1)
      newFile = r.TFile('/vols/cms/es811/Stage1categorisation/trainTrees/new.root','RECREATE')
      newTree = trainTree.CloneTree()
      trainFrames[proc] = pd.DataFrame( tree2array(newTree) )
      del newTree
      del newFile
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
  
  #some extra cuts that are applied for diphoton BDT in the AN
  trainTotal = trainTotal[trainTotal.leadmva>-0.2]
  trainTotal = trainTotal[trainTotal.subleadmva>-0.2]
  trainTotal = trainTotal[trainTotal.leadptom>0.333]
  trainTotal = trainTotal[trainTotal.subleadptom>0.25]
  trainTotal = trainTotal[trainTotal.stage1cat>-1.]
  print 'done basic preselection cuts'

  #apply VBF preselection
  trainTotal = trainTotal[trainTotal.dijet_Mjj>250.]
  trainTotal = trainTotal[trainTotal.dijet_LeadJPt>40.]
  trainTotal = trainTotal[trainTotal.dijet_SubJPt>30.]
  
  #select VBF and background only
  bkgTotal = trainTotal[trainTotal.stage1cat==0]
  vbfTotal = trainTotal[trainTotal.stage1cat>11]
  vbfTotal = vbfTotal[vbfTotal.stage1cat<17]
  trainTotal = pd.concat([bkgTotal,vbfTotal])
  del bkgTotal
  del vbfTotal
  
  #add extra info to dataframe
  print 'about to add extra columns'
  trainTotal['combinedWeight'] = trainTotal.apply(combinedWeight,axis=1)
  print 'all columns added'

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/combinedTotal.pkl'%frameDir)
  print 'frame saved as %s/combinedTotal.pkl'%frameDir
else:
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
  print 'Successfully loaded the dataframe'

#define the variables used as input to the classifier
diphoTW = trainTotal['combinedWeight'].values
diphoFW = trainTotal['weight'].values
diphoP  = trainTotal['stage1cat'].values
diphoY  = np.ones(diphoP.shape[0]) * (diphoP>1)

# either use what is already packaged up or perform inference for diphoton BDT
if not opts.modelName:
  combinedVars = ['VBFMVAValue','diphomvaxgb','leadptom','subleadptom']
  diphoX  = trainTotal[combinedVars].values
else:
  diphoX = trainTotal[diphoVars].values
  diphoMatrix = xg.DMatrix(diphoX, label=diphoP, weight=diphoFW, feature_names=diphoVars)
  diphoModel = xg.Booster()
  diphoModel.load_model('%s/%s'%(modelDir,opts.modelName))
  trainTotal['diphomvainferred'] = diphoModel.predict(diphoMatrix)
  combinedVars = ['VBFMVAValue','diphomvainferred','leadptom','subleadptom']
  diphoX  = trainTotal[combinedVars].values

#define the indices shuffle
theShape = trainTotal.shape[0]
diphoShuffle = np.random.permutation(theShape)
diphoTrainLimit = int(theShape*trainFrac)
diphoValidLimit = int(theShape*(trainFrac+validFrac))

#shuffle
diphoX  = diphoX[diphoShuffle]
diphoTW = diphoTW[diphoShuffle]
diphoFW = diphoFW[diphoShuffle]
diphoY  = diphoY[diphoShuffle]

#split
diphoTrainX,  diphoValidX,  diphoTestX  = np.split( diphoX,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainTW, diphoValidTW, diphoTestTW = np.split( diphoTW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainFW, diphoValidFW, diphoTestFW = np.split( diphoFW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainY,  diphoValidY,  diphoTestY  = np.split( diphoY,  [diphoTrainLimit,diphoValidLimit] )

#build the combined BDT
trainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainTW, feature_names=combinedVars)
testingDipho  = xg.DMatrix(diphoTestX,  label=diphoTestY,  weight=diphoTestFW,  feature_names=combinedVars)
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
print 'about to train combined BDT'
combinedModel = xg.train(trainParams, trainingDipho)
print 'done'

#save it
modelExt = ''
if opts.modelName:
  modelExt = '__%s'%opts.modelName.replace('.model','')
combinedModel.save_model('%s/combinedModel%s%s.model'%(modelDir,modelExt,paramExt))
print 'saved as %s/combinedModel%s%s.model'%(modelDir,modelExt,paramExt)

#check performance of each training
combinedPredYxcheck = combinedModel.predict(trainingDipho)
combinedPredY       = combinedModel.predict(testingDipho)
print 'Default training performance:'
print 'area under roc curve for training set = %1.3f'%( roc_auc_score(diphoTrainY, combinedPredYxcheck, sample_weight=diphoTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( roc_auc_score(diphoTestY,  combinedPredY,       sample_weight=diphoTestFW) )
