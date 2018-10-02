#usual imports
import ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system

from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Name of dataframe if it already exists')
parser.add_option('-s','--signifFrame', default=None, help='Name of cleaned signal dataframe if it already exists')
parser.add_option('-m','--modelName', default=None, help='Name of model for testing')
parser.add_option('-n','--nIterations', default=1000, help='Number of iterations to run for random significance optimisation')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models')
nClasses = 9

#get trees from files, put them in data frames
procFileMap = {'Data':'Data.root'}
theProcs = procFileMap.keys()

#define the different sets of variables used
diphoVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      trainFile   = r.TFile('%s/%s'%(trainDir,fn))
      if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
      else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
      trainTree.SetBranchStatus('nvtx',0)
      trainTree.SetBranchStatus('VBFMVAValue',0)
      trainTree.SetBranchStatus('dijet_*',0)
      trainTree.SetBranchStatus('dZ',0)
      trainTree.SetBranchStatus('centralObjectWeight',0)
      trainTree.SetBranchStatus('rho',0)
      trainTree.SetBranchStatus('nvtx',0)
      trainTree.SetBranchStatus('event',0)
      trainTree.SetBranchStatus('lumi',0)
      trainTree.SetBranchStatus('processIndex',0)
      trainTree.SetBranchStatus('run',0)
      trainTree.SetBranchStatus('npu',0)
      trainTree.SetBranchStatus('puweight',0)
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
  dataTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  #then filter out the events into only those with the phase space we are interested in
  dataTotal = dataTotal[dataTotal.CMS_hgg_mass>100.]
  dataTotal = dataTotal[dataTotal.CMS_hgg_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for diphoton BDT in the AN
  dataTotal = dataTotal[dataTotal.leadmva>-0.9]
  dataTotal = dataTotal[dataTotal.subleadmva>-0.9]
  dataTotal = dataTotal[dataTotal.leadptom>0.333]
  dataTotal = dataTotal[dataTotal.subleadptom>0.25]
  print 'done basic preselection cuts'
  
  #add extra info to dataframe
  print 'about to add extra columns'
  dataTotal['diphopt'] = dataTotal.apply(addPt, axis=1)
  dataTotal['reco'] = dataTotal.apply(reco,axis=1)
  print 'all columns added'

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  dataTotal.to_pickle('%s/dataTotal.pkl'%frameDir)
  print 'frame saved as %s/dataTotal.pkl'%frameDir

#read in dataframe if above steps done before
else:
  dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
  print 'Successfully loaded the dataframe'


if not opts.signifFrame:
  #read in signal mc dataframe
  trainTotal = pd.read_pickle('%s/trainTotal.pkl'%frameDir)
  print 'Successfully loaded the signal dataframe'
  
  #remove bkg then add reco tag info
  trainTotal = trainTotal[trainTotal.stage1cat>0.01]
  trainTotal = trainTotal[trainTotal.stage1cat<12.]
  trainTotal = trainTotal[trainTotal.stage1cat!=1]
  trainTotal = trainTotal[trainTotal.stage1cat!=2]
  print 'About to add reco tag info'
  trainTotal['diphopt'] = trainTotal.apply(addPt, axis=1)
  trainTotal['reco'] = trainTotal.apply(reco, axis=1)
  print 'Successfully added reco tag info'

  #save
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/signifTotal.pkl'%frameDir)
  print 'frame saved as %s/signifTotal.pkl'%frameDir

else:
  #read in already cleaned up signal frame
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.signifFrame))

#define the variables used as input to the classifier
diphoX  = trainTotal[diphoVars].values
diphoY  = trainTotal['truthDipho'].values
diphoFW = trainTotal['weight'].values
diphoS  = trainTotal['stage1cat'].values
diphoR  = trainTotal['reco'].values
diphoM  = trainTotal['CMS_hgg_mass'].values

dataX  = dataTotal[diphoVars].values
dataY  = np.zeros(dataX.shape[0])
dataFW = np.ones(dataX.shape[0])
dataR  = dataTotal['reco'].values
dataM  = dataTotal['CMS_hgg_mass'].values

#setup matrices
diphoMatrix = xg.DMatrix(diphoX, label=diphoY, weight=diphoFW, feature_names=diphoVars)
dataMatrix  = xg.DMatrix(dataX,  label=dataY,  weight=dataFW,  feature_names=diphoVars)

#loadi the model to be tested
diphoModel = xg.Booster()
diphoModel.load_model('%s/%s'%(modelDir,opts.modelName))

#get predicted values
diphoPredY = diphoModel.predict(diphoMatrix)
dataPredY  = diphoModel.predict(dataMatrix)

#now estimate two-class significance
printStr = ''
lumi = opts.intLumi
for iProc in range(nClasses):
  bestCutLo = -.1
  bestCutHi = -.1
  bestSignifLo = -1.
  bestSignifHi = -1.
  bestSlo = -1.
  bestShi = -1.
  bestBlo = -1.
  bestBhi = -1.
  bestSignif = -1.
  for i in range(opts.nIterations): 
    cuts = np.random.uniform(0.,1.,2)
    cuts.sort()
    cutLo = cuts[0]
    cutHi = cuts[1]
    sigHistHi = r.TH1F('sigHistHiTemp','sigHistHiTemp',160,100,180)
    sigWeightsHi = diphoFW * (diphoY==1) * (diphoS-3==iProc) * (diphoPredY>cutHi) * (diphoR==iProc)
    fill_hist(sigHistHi, diphoM, weights=sigWeightsHi)
    sigCountHi = 0.68 * lumi * sigHistHi.Integral() 
    sigWidthHi = getRealSigma(sigHistHi)
    bkgHistHi = r.TH1F('bkgHistHiTemp','bkgHistHiTemp',160,100,180)
    bkgWeightsHi = dataFW * (dataPredY>cutHi) * (dataR==iProc)
    fill_hist(bkgHistHi, dataM, weights=bkgWeightsHi)
    bkgCountHi = computeBkg(bkgHistHi, sigWidthHi)
    theSignifHi = getAMS(sigCountHi, bkgCountHi)
    sigHistLo = r.TH1F('sigHistLoTemp','sigHistLoTemp',160,100,180)
    sigWeightsLo = diphoFW * (diphoY==1) * (diphoS-3==iProc) * (diphoPredY<cutHi) * (diphoPredY>cutLo) * (diphoR==iProc)
    fill_hist(sigHistLo, diphoM, weights=sigWeightsLo)
    sigCountLo = 0.68 * lumi * sigHistLo.Integral() 
    sigWidthLo = getRealSigma(sigHistLo)
    #print 'sigwidth is %1.3f'%sigWidth
    bkgHistLo = r.TH1F('bkgHistLoTemp','bkgHistLoTemp',160,100,180)
    bkgWeightsLo = dataFW * (dataPredY<cutHi) * (dataPredY>cutLo) * (dataR==iProc)
    fill_hist(bkgHistLo, dataM, weights=bkgWeightsLo)
    bkgCountLo = computeBkg(bkgHistLo, sigWidthLo)
    theSignifLo = getAMS(sigCountLo, bkgCountLo)
    theSignif = np.sqrt( theSignifLo*theSignifLo + theSignifHi*theSignifHi )
    if theSignif > bestSignif: 
      bestCutLo = cutLo
      bestCutHi = cutHi
      bestSignif = theSignif
      bestSignifLo = theSignifLo
      bestSignifHi = theSignifHi
      bestSlo = sigCountLo
      bestBlo = bkgCountLo
      bestShi = sigCountHi
      bestBhi = bkgCountHi
  printStr += 'for proc %g the best outcome was tot signif %1.2f:\n'%(iProc,bestSignif)
  printStr += 'cutHi %1.3f, Shi %1.2f, Bhi %1.2f, signifHi %1.2f \n'%(bestCutHi,bestShi,bestBhi,bestSignifHi)
  printStr += 'cutLo %1.3f, Slo %1.2f, Blo %1.2f, signifLo %1.2f \n'%(bestCutLo,bestSlo,bestBlo,bestSignifLo)
  printStr += '\n'

print printStr
