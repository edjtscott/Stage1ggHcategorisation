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
from catOptim import CatOptim

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Name of dataframe if it already exists')
parser.add_option('-s','--signalFrame', default=None, help='Name of signal dataframe if it already exists')
parser.add_option('-m','--modelName', default=None, help='Name of model for testing')
parser.add_option('-n','--nIterations', default=10000, help='Number of iterations to run for random significance optimisation')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models').replace('ForVBF/','')

#define the different sets of variables used
diphoVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

#get trees from files, put them in data frames
#procFileMap = {'vbf':'VBF.root'}
procFileMap = {'ggh':'ggH.root','vbf':'VBF.root'}
theProcs = procFileMap.keys()
dataFileMap = {'Data':'Data.root'}

#read or construct dataframes
trainTotal = None
if not opts.signalFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
    trainFile   = r.TFile('%s/%s'%(trainDir,fn))
    if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
    else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
    trainTree.SetBranchStatus('nvtx',0)
    trainTree.SetBranchStatus('dijet_*',0)
    trainTree.SetBranchStatus('dijet_Mjj',1)
    trainTree.SetBranchStatus('dijet_LeadJPt',1)
    trainTree.SetBranchStatus('dijet_SubJPt',1)
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
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  #then filter out the events into only those with the phase space we are interested in
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass>100.]
  trainTotal = trainTotal[trainTotal.CMS_hgg_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for diphoton BDT in the AN
  #trainTotal = trainTotal[trainTotal.leadmva>-0.9]
  #trainTotal = trainTotal[trainTotal.subleadmva>-0.9] #FIXME try changing these as in paper
  trainTotal = trainTotal[trainTotal.leadmva>-0.2]
  trainTotal = trainTotal[trainTotal.subleadmva>-0.2]
  trainTotal = trainTotal[trainTotal.leadptom>0.333]
  trainTotal = trainTotal[trainTotal.subleadptom>0.25]
  trainTotal = trainTotal[trainTotal.dijet_Mjj>250.]
  trainTotal = trainTotal[trainTotal.dijet_LeadJPt>40.]
  trainTotal = trainTotal[trainTotal.dijet_SubJPt>30.]
  print 'done VBF preselection cuts'

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/vbfTotal.pkl'%frameDir)
  print 'frame saved as %s/vbfTotal.pkl'%frameDir
else:
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.signalFrame))

dataTotal = None
if not opts.dataFrame:
  dataFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in dataFileMap.iteritems():
    trainFile   = r.TFile('%s/%s'%(trainDir,fn))
    if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
    else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
    trainTree.SetBranchStatus('nvtx',0)
    trainTree.SetBranchStatus('dijet_*',0)
    trainTree.SetBranchStatus('dijet_Mjj',1)
    trainTree.SetBranchStatus('dijet_LeadJPt',1)
    trainTree.SetBranchStatus('dijet_SubJPt',1)
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
    dataFrames[proc] = pd.DataFrame( tree2array(newTree) )
    del newTree
    del newFile
    dataFrames[proc]['proc'] = proc
  print 'got trees'

  dataTotal = dataFrames['Data']
  
  #then filter out the events into only those with the phase space we are interested in
  dataTotal = dataTotal[dataTotal.CMS_hgg_mass>100.]
  dataTotal = dataTotal[dataTotal.CMS_hgg_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for diphoton BDT in the AN
  #dataTotal = dataTotal[dataTotal.leadmva>-0.9]
  #dataTotal = dataTotal[dataTotal.subleadmva>-0.9] #FIXME
  dataTotal = dataTotal[dataTotal.leadmva>-0.2]
  dataTotal = dataTotal[dataTotal.subleadmva>-0.2]
  dataTotal = dataTotal[dataTotal.leadptom>0.333]
  dataTotal = dataTotal[dataTotal.subleadptom>0.25]
  dataTotal = dataTotal[dataTotal.dijet_Mjj>250.]
  dataTotal = dataTotal[dataTotal.dijet_LeadJPt>40.]
  dataTotal = dataTotal[dataTotal.dijet_SubJPt>30.]
  print 'done VBF preselection cuts'

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  dataTotal.to_pickle('%s/dataTotal.pkl'%frameDir)
  print 'frame saved as %s/dataTotal.pkl'%frameDir
else:
  dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))

#define the variables used as input to the classifier
diphoM  = trainTotal['CMS_hgg_mass'].values
diphoFW = trainTotal['weight'].values
diphoP  = trainTotal['stage1cat'].values
diphoV  = trainTotal['VBFMVAValue'].values
diphoH  = trainTotal['ptHjj'].values
diphoJ  = trainTotal['dijet_Mjj'].values

dataM  = dataTotal['CMS_hgg_mass'].values
dataFW = np.ones(dataM.shape[0])
dataV  = dataTotal['VBFMVAValue'].values
dataH  = dataTotal['ptHjj'].values
dataJ  = dataTotal['dijet_Mjj'].values

# either use what is already packaged up or perform inference
if not opts.modelName:
  diphoD  = trainTotal['diphomvaxgb'].values
  dataD  = dataTotal['diphomvaxgb'].values
else:
  diphoX = trainTotal[diphoVars].values
  dataX  = dataTotal[diphoVars].values
  diphoMatrix = xg.DMatrix(diphoX, label=diphoP, weight=diphoFW, feature_names=diphoVars)
  dataMatrix  = xg.DMatrix(dataX,  label=dataFW, weight=dataFW,  feature_names=diphoVars)
  diphoModel = xg.Booster()
  diphoModel.load_model('%s/%s'%(modelDir,opts.modelName))
  diphoD = diphoModel.predict(diphoMatrix)
  dataD  = diphoModel.predict(dataMatrix)

#now estimate significance using the amount of background in a plus/mins 1 sigma window
#set up parameters for the optimiser
ptHjjCut = 25.
ranges = [ [-0.9,1.], [0.5,1.] ]
names  = ['DijetBDT','DiphotonBDT']
printStr = ''

#first the low ptHjj bin
sigWeights = diphoFW * (diphoP>11) * (diphoP<17) * (diphoH<ptHjjCut)
bkgWeights = dataFW * (dataH<ptHjjCut)
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for low ptHjj bin are: \n'
printStr += optimiser.getPrintableResult()

#first the low ptHjj bin
sigWeights = diphoFW * (diphoP>11) * (diphoP<17) * (diphoH<ptHjjCut)
bkgWeights = dataFW * (dataH<ptHjjCut)
nonSigWeights = diphoFW * (diphoP<12) * (diphoP>2) * (diphoH<ptHjjCut)
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.setNonSig(nonSigWeights, diphoM, [diphoV,diphoD])
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for low ptHjj bin are: \n'
printStr += optimiser.getPrintableResult()

#and now the high bin
sigWeights = diphoFW * (diphoP>11) * (diphoP<17) * (diphoH>ptHjjCut)
bkgWeights = dataFW * (dataH>ptHjjCut)
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for high ptHjj bin are: \n'
printStr += optimiser.getPrintableResult()

#repeat with no ptHjj cut
sigWeights = diphoFW * (diphoP>11) * (diphoP<17)
bkgWeights = dataFW
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results with no ptHjj cut are: \n'
printStr += optimiser.getPrintableResult()

#repeat with no ptHjj cut, three-cat version
sigWeights = diphoFW * (diphoP>11) * (diphoP<17)
bkgWeights = dataFW
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 3, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results with no ptHjj cut are: \n'
printStr += optimiser.getPrintableResult()

#all three again with mjj cut increased to 400
sigWeights = diphoFW * (diphoP>11) * (diphoP<17) * (diphoH<ptHjjCut) * (diphoJ>400)
bkgWeights = dataFW * (dataH<ptHjjCut) * (dataJ>400)
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for low ptHjj bin with mjj over 400 are: \n'
printStr += optimiser.getPrintableResult()

#and now the high bin
sigWeights = diphoFW * (diphoP>11) * (diphoP<17) * (diphoH>ptHjjCut) * (diphoJ>400)
bkgWeights = dataFW * (dataH>ptHjjCut) * (dataJ>400)
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for high ptHjj bin with mjj over 400 are: \n'
printStr += optimiser.getPrintableResult()

#no ptHjj cut and mjj > 400
sigWeights = diphoFW * (diphoP>11) * (diphoP<17) * (diphoJ>400)
bkgWeights = dataFW * (dataJ>400)
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results with no ptHjj cut and mjj over 400 are: \n'
printStr += optimiser.getPrintableResult()

#no ptHjj cut and mjj > 400
sigWeights = diphoFW * (diphoP>11) * (diphoP<17) * (diphoJ>400)
bkgWeights = dataFW * (dataJ>400)
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 3, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results with no ptHjj cut and mjj over 400 are: \n'
printStr += optimiser.getPrintableResult()

# test a version targeting "VBF rest" bin - one cat only
sigWeights = diphoFW * (diphoP>11) * (diphoP<17) * (diphoJ>250) * (diphoJ<400)
bkgWeights = dataFW * (dataJ>250) * (dataJ<400)
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 1, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for the VBF rest bin are: \n'
printStr += optimiser.getPrintableResult()

# test a version targeting "VBF rest" bin - two cats
sigWeights = diphoFW * (diphoP>11) * (diphoP<17) * (diphoJ>250) * (diphoJ<400)
bkgWeights = dataFW * (dataJ>250) * (dataJ<400)
optimiser = CatOptim(sigWeights, diphoM, [diphoV,diphoD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for the VBF rest bin are: \n'
printStr += optimiser.getPrintableResult()

print
print printStr
print
