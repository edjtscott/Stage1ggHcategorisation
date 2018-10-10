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
parser.add_option('-s','--signalFrame', default=None, help='Name of signal dataframe if it already exists')
parser.add_option('-m','--modelName', default=None, help='Name of model for testing')
parser.add_option('-n','--nIterations', default=5000, help='Number of iterations to run for random significance optimisation')
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
procFileMap = {'vbf':'VBF.root'}
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
  trainTotal = trainTotal[trainTotal.leadmva>-0.9]
  trainTotal = trainTotal[trainTotal.subleadmva>-0.9]
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
  dataTotal = dataTotal[dataTotal.leadmva>-0.9]
  dataTotal = dataTotal[dataTotal.subleadmva>-0.9]
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

dataM  = dataTotal['CMS_hgg_mass'].values
dataFW = np.ones(dataM.shape[0])
dataV  = dataTotal['VBFMVAValue'].values
dataH  = dataTotal['ptHjj'].values

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
lumi = opts.intLumi
printStr = ''

#first the low ptHjj bin
ptHjjCut = 25.
bestCutLo = -.1
bestSignifLo = -1.
bestSLo = -1.
bestBLo = -1.
bestCutHi = -.1
bestSignifHi = -1.
bestSHi = -1.
bestBHi = -1.
bestSignif = -1.
bestDiphoCutLo = -.1
bestDiphoCutHi = -.1
for i in range(opts.nIterations):
  if i%100==0: print 'processing iteration %g'%i
  cuts = np.random.uniform(-1., 1., 2)
  diphoCuts = np.random.uniform(0.6, 1., 2)
  cuts.sort()
  diphoCuts.sort()
  lowCut  = cuts[0]
  highCut = cuts[1]
  diphoCutLo = diphoCuts[0]
  diphoCutHi = diphoCuts[1]
  sigHistHi = r.TH1F('sigHistHiTemp','sigHistHiTemp',160,100,180)
  sigWeightsHi = diphoFW * (diphoP>11) * (diphoP<17) * (diphoV>highCut) * (diphoD>diphoCutHi) * (diphoH<ptHjjCut)
  fill_hist(sigHistHi, diphoM, weights=sigWeightsHi)
  sigCountHi = 0.68 * lumi * sigHistHi.Integral() 
  sigWidthHi = getRealSigma(sigHistHi)
  bkgHistHi = r.TH1F('bkgHistHiTemp','bkgHistHiTemp',160,100,180)
  bkgWeightsHi = dataFW * (dataV>highCut) * (dataD>diphoCutHi) * (dataH<ptHjjCut)
  fill_hist(bkgHistHi, dataM, weights=bkgWeightsHi)
  bkgCountHi = computeBkg(bkgHistHi, sigWidthHi)
  theSignifHi = getAMS(sigCountHi, bkgCountHi)
  sigHistLo = r.TH1F('sigHistLoTemp','sigHistLoTemp',160,100,180)
  sigWeightsLo = diphoFW * (diphoP>11) * (diphoP<17) * (diphoV<highCut) * (diphoV>lowCut) * (diphoD>diphoCutLo) * (diphoH<ptHjjCut)
  fill_hist(sigHistLo, diphoM, weights=sigWeightsLo)
  sigCountLo = 0.68 * lumi * sigHistLo.Integral()
  sigWidthLo = getRealSigma(sigHistLo)
  bkgHistLo = r.TH1F('bkgHistLoTemp','bkgHistLoTemp',160,100,180)
  bkgWeightsLo = dataFW * (dataV>lowCut) * (dataV<highCut) * (dataD>diphoCutLo) * (dataH<ptHjjCut)
  fill_hist(bkgHistLo, dataM, weights=bkgWeightsLo)
  bkgCountLo = computeBkg(bkgHistLo, sigWidthLo)
  theSignifLo = getAMS(sigCountLo, bkgCountLo)
  theSignif = np.sqrt(theSignifLo*theSignifLo + theSignifHi*theSignifHi)
  if theSignif > bestSignif: 
    bestCutLo = lowCut
    bestSignifLo = theSignifLo
    bestSLo = sigCountLo
    bestBLo = bkgCountLo
    bestCutHi = highCut
    bestSignifHi = theSignifHi
    bestSHi = sigCountHi
    bestBHi = bkgCountHi
    bestSignif = theSignif
    bestDiphoCutLo = diphoCutLo
    bestDiphoCutHi = diphoCutHi
printStr +=  'total significance for low ptHjj bin is %1.3f\n'%bestSignif
printStr +=  'cutHi %1.3f, diphoCutHi %1.3f, Shi %1.2f, Bhi %1.2f, signifHi %1.2f\n'%(bestCutHi,diphoCutHi,bestSHi,bestBHi,bestSignifHi)
printStr +=  'cutLo %1.3f, diphoCutLo %1.3f, Slo %1.2f, Blo %1.2f, signifLo %1.2f\n'%(bestCutLo,diphoCutLo,bestSLo,bestBLo,bestSignifLo)
printStr += '\n'

#and now the high bin
bestCutLo = -.1
bestSignifLo = -1.
bestSLo = -1.
bestBLo = -1.
bestCutHi = -.1
bestSignifHi = -1.
bestSHi = -1.
bestBHi = -1.
bestSignif = -1.
bestDiphoCutLo = -.1
bestDiphoCutHi = -.1
for i in range(opts.nIterations):
  if i%100==0: print 'processing iteration %g'%i
  cuts = np.random.uniform(-1., 1., 2)
  diphoCuts = np.random.uniform(0.6, 1., 2)
  cuts.sort()
  diphoCuts.sort()
  lowCut  = cuts[0]
  highCut = cuts[1]
  diphoCutLo = diphoCuts[0]
  diphoCutHi = diphoCuts[1]
  sigHistHi = r.TH1F('sigHistHiTemp','sigHistHiTemp',160,100,180)
  sigWeightsHi = diphoFW * (diphoP>11) * (diphoP<17) * (diphoV>highCut) * (diphoD>diphoCutHi) * (diphoH>ptHjjCut)
  fill_hist(sigHistHi, diphoM, weights=sigWeightsHi)
  sigCountHi = 0.68 * lumi * sigHistHi.Integral() 
  sigWidthHi = getRealSigma(sigHistHi)
  bkgHistHi = r.TH1F('bkgHistHiTemp','bkgHistHiTemp',160,100,180)
  bkgWeightsHi = dataFW * (dataV>highCut) * (dataD>diphoCutHi) * (dataH>ptHjjCut)
  fill_hist(bkgHistHi, dataM, weights=bkgWeightsHi)
  bkgCountHi = computeBkg(bkgHistHi, sigWidthHi)
  theSignifHi = getAMS(sigCountHi, bkgCountHi)
  sigHistLo = r.TH1F('sigHistLoTemp','sigHistLoTemp',160,100,180)
  sigWeightsLo = diphoFW * (diphoP>11) * (diphoP<17) * (diphoV<highCut) * (diphoV>lowCut) * (diphoD>diphoCutLo) * (diphoH>ptHjjCut)
  fill_hist(sigHistLo, diphoM, weights=sigWeightsLo)
  sigCountLo = 0.68 * lumi * sigHistLo.Integral()
  sigWidthLo = getRealSigma(sigHistLo)
  bkgHistLo = r.TH1F('bkgHistLoTemp','bkgHistLoTemp',160,100,180)
  bkgWeightsLo = dataFW * (dataV>lowCut) * (dataV<highCut) * (dataD>diphoCutLo) * (dataH>ptHjjCut)
  fill_hist(bkgHistLo, dataM, weights=bkgWeightsLo)
  bkgCountLo = computeBkg(bkgHistLo, sigWidthLo)
  theSignifLo = getAMS(sigCountLo, bkgCountLo)
  theSignif = np.sqrt(theSignifLo*theSignifLo + theSignifHi*theSignifHi)
  if theSignif > bestSignif: 
    bestCutLo = lowCut
    bestSignifLo = theSignifLo
    bestSLo = sigCountLo
    bestBLo = bkgCountLo
    bestCutHi = highCut
    bestSignifHi = theSignifHi
    bestSHi = sigCountHi
    bestBHi = bkgCountHi
    bestSignif = theSignif
    bestDiphoCutLo = diphoCutLo
    bestDiphoCutHi = diphoCutHi
printStr +=  'total significance for high ptHjj bin is %1.3f\n'%bestSignif
printStr +=  'cutHi %1.3f, diphoCutHi %1.3f, Shi %1.2f, Bhi %1.2f, signifHi %1.2f\n'%(bestCutHi,diphoCutHi,bestSHi,bestBHi,bestSignifHi)
printStr +=  'cutLo %1.3f, diphoCutLo %1.3f, Slo %1.2f, Blo %1.2f, signifLo %1.2f\n'%(bestCutLo,diphoCutLo,bestSLo,bestBLo,bestSignifLo)
printStr += '\n'

#repeat with no ptHjj cut
bestCutLo = -.1
bestSignifLo = -1.
bestSLo = -1.
bestBLo = -1.
bestCutHi = -.1
bestSignifHi = -1.
bestSHi = -1.
bestBHi = -1.
bestSignif = -1.
bestDiphoCutLo = -.1
bestDiphoCutHi = -.1
for i in range(opts.nIterations):
  if i%100==0: print 'processing iteration %g'%i
  cuts = np.random.uniform(-1., 1., 2)
  diphoCuts = np.random.uniform(0.6, 1., 2)
  cuts.sort()
  diphoCuts.sort()
  lowCut  = cuts[0]
  highCut = cuts[1]
  diphoCutLo = diphoCuts[0]
  diphoCutHi = diphoCuts[1]
  sigHistHi = r.TH1F('sigHistHiTemp','sigHistHiTemp',160,100,180)
  sigWeightsHi = diphoFW * (diphoP>11) * (diphoP<17) * (diphoV>highCut) * (diphoD>diphoCutHi)
  fill_hist(sigHistHi, diphoM, weights=sigWeightsHi)
  sigCountHi = 0.68 * lumi * sigHistHi.Integral() 
  sigWidthHi = getRealSigma(sigHistHi)
  bkgHistHi = r.TH1F('bkgHistHiTemp','bkgHistHiTemp',160,100,180)
  bkgWeightsHi = dataFW * (dataV>highCut) * (dataD>diphoCutHi)
  fill_hist(bkgHistHi, dataM, weights=bkgWeightsHi)
  bkgCountHi = computeBkg(bkgHistHi, sigWidthHi)
  theSignifHi = getAMS(sigCountHi, bkgCountHi)
  sigHistLo = r.TH1F('sigHistLoTemp','sigHistLoTemp',160,100,180)
  sigWeightsLo = diphoFW * (diphoP>11) * (diphoP<17) * (diphoV<highCut) * (diphoV>lowCut) * (diphoD>diphoCutLo)
  fill_hist(sigHistLo, diphoM, weights=sigWeightsLo)
  sigCountLo = 0.68 * lumi * sigHistLo.Integral()
  sigWidthLo = getRealSigma(sigHistLo)
  bkgHistLo = r.TH1F('bkgHistLoTemp','bkgHistLoTemp',160,100,180)
  bkgWeightsLo = dataFW * (dataV>lowCut) * (dataV<highCut) * (dataD>diphoCutLo)
  fill_hist(bkgHistLo, dataM, weights=bkgWeightsLo)
  bkgCountLo = computeBkg(bkgHistLo, sigWidthLo)
  theSignifLo = getAMS(sigCountLo, bkgCountLo)
  theSignif = np.sqrt(theSignifLo*theSignifLo + theSignifHi*theSignifHi)
  if theSignif > bestSignif: 
    bestCutLo = lowCut
    bestSignifLo = theSignifLo
    bestSLo = sigCountLo
    bestBLo = bkgCountLo
    bestCutHi = highCut
    bestSignifHi = theSignifHi
    bestSHi = sigCountHi
    bestBHi = bkgCountHi
    bestSignif = theSignif
    bestDiphoCutLo = diphoCutLo
    bestDiphoCutHi = diphoCutHi
printStr +=  'total significance with no ptHjj cut is %1.3f\n'%bestSignif
printStr +=  'cutHi %1.3f, diphoCutHi %1.3f, Shi %1.2f, Bhi %1.2f, signifHi %1.2f\n'%(bestCutHi,diphoCutHi,bestSHi,bestBHi,bestSignifHi)
printStr +=  'cutLo %1.3f, diphoCutLo %1.3f, Slo %1.2f, Blo %1.2f, signifLo %1.2f\n'%(bestCutLo,diphoCutLo,bestSLo,bestBLo,bestSignifLo)
printStr += '\n'

print printStr
