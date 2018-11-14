#usual imports
import ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system

from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthJets, jetWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams',default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
#parser.add_option('--equalWeights', default=False, action='store_true', help='Alter weights for training so that signal and background have equal sum of weights')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')
trainFrac = 0.7
validFrac = 0.1
nJetClasses = 3
nClasses = 9
binNames = ['0J', '1J low', '1J med', '1J high', '1J BSM', '2J low', '2J med', '2J high', '2J BSM']

#define the different sets of variables used
diphoVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

classVars  = ['n_rec_jets','dijet_Mjj','diphopt',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta']

jetVars  = ['n_rec_jets','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta']

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root'}
theProcs = procFileMap.keys()

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
  trainTotal = trainTotal[trainTotal.stage1cat>-1.]
  print 'done basic preselection cuts'
  
  #add extra info to dataframe
  print 'about to add extra columns'
  trainTotal['diphopt'] = trainTotal.apply(addPt, axis=1)
  trainTotal['reco'] = trainTotal.apply(reco, axis=1)
  trainTotal['truthJets'] = trainTotal.apply(truthJets, axis=1)
  trainTotal['jetWeight'] = trainTotal.apply(jetWeight, axis=1)
  print 'all columns added'

  #only select processes relevant for nJet training
  trainTotal = trainTotal[trainTotal.truthJets>-1]

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/jetTotal.pkl'%frameDir)
  print 'frame saved as %s/jetTotal.pkl'%frameDir

#read in dataframe if above steps done before
else:
  trainTotal = pd.read_pickle(opts.dataFrame)
  print 'Successfully loaded the dataframe'

#shape and shuffle definitions
theShape = trainTotal.shape[0]
classShuffle = np.random.permutation(theShape)
classTrainLimit = int(theShape*trainFrac)
classValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for multiclass training
classI  = trainTotal[jetVars].values
classJ  = trainTotal['truthJets'].values
classJW  = trainTotal['jetWeight'].values
classFW  = trainTotal['weight'].values
classR  = trainTotal['reco'].values
classM  = trainTotal['CMS_hgg_mass'].values

classI  = classI[classShuffle]
classJ  = classJ[classShuffle]
classJW = classJW[classShuffle]
classFW = classFW[classShuffle]
classR  = classR[classShuffle]
classM  = classM[classShuffle]

classTrainI,  classValidI,  classTestI  = np.split( classI,  [classTrainLimit,classValidLimit] )
classTrainJ,  classValidJ,  classTestJ  = np.split( classJ,  [classTrainLimit,classValidLimit] )
classTrainJW,  classValidJW,  classTestJW  = np.split( classJW,  [classTrainLimit,classValidLimit] )
classTrainFW,  classValidFW,  classTestFW  = np.split( classFW,  [classTrainLimit,classValidLimit] )
classTrainR,  classValidR,  classTestR  = np.split( classR,  [classTrainLimit,classValidLimit] )
classTrainM,  classValidM,  classTestM  = np.split( classM,  [classTrainLimit,classValidLimit] )

#build the jet-classifier
trainingJet = xg.DMatrix(classTrainI, label=classTrainJ, weight=classTrainJW, feature_names=jetVars)
testingJet  = xg.DMatrix(classTestI,  label=classTestJ,  weight=classTestFW,  feature_names=jetVars)
trainParams = {}
trainParams['objective'] = 'multi:softprob'
trainParams['num_class'] = nJetClasses
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
print 'about to train jet counting BDT'
jetModel = xg.train(trainParams, trainingJet)
print 'done'

#save
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
jetModel.save_model('%s/jetModel%s.model'%(modelDir,paramExt))
print 'saved as %s/jetModel%s.model'%(modelDir,paramExt)

exit('stop here for now')

#get predicted values
import otherHelpers
reload(otherHelpers)
predProbJet = jetModel.predict(testingJet).reshape(classTestJ.shape[0],nJetClasses) #FIXME: not sure if this is right
#priors = np.array( [0.2994, 0.0757, 0.0530, 0.0099, 0.0029, 0.0154, 0.0235, 0.0165, 0.0104] )
priors = np.array( [0.2994, (0.0757+0.0530+0.0099+0.0029), (0.0154+0.0235+0.0165+0.0104)] )
priors *= 1. / priors.sum()
predProbJet *= priors
classPredJ = np.argmax(predProbJet, axis=1)
classPredY = otherHelpers.jetPtToClass(classPredJ, classTestP)

print classPredJ
print classTestP
print
print classTestY
print classPredY
print

#define reco hists
canv = useSty.setCanvas()
truthHist = r.TH1F('truthHist','truthHist',nClasses,-0.5,nClasses-0.5)
truthHist.SetTitle('')
useSty.formatHisto(truthHist)
predHist  = r.TH1F('predHist','predHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(predHist)
predHist.SetTitle('')
rightHist = r.TH1F('rightHist','rightHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(rightHist)
rightHist.SetTitle('')
wrongHist = r.TH1F('wrongHist','wrongHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(wrongHist)
wrongHist.SetTitle('')

for iBin in range(1, nClasses+1):
    truthHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    predHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    rightHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    wrongHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])

for true,guess,w in zip(classTestY,classTestR,classTestFW):
    truthHist.Fill(true,w)
    predHist.Fill(guess,w)
    if true==guess: rightHist.Fill(true,w)
    else: wrongHist.Fill(true,w)
        
firstBinVal = -1.
for iBin in range(1,truthHist.GetNbinsX()+1):
    if iBin==1: firstBinVal = truthHist.GetBinContent(iBin)
    ratio = float(truthHist.GetBinContent(iBin)) / firstBinVal
    print 'ratio for bin %g is %1.7f'%(iBin,ratio)
    
wrongHist.Add(rightHist)
rightHist.Divide(wrongHist)
effHist = r.TH1F
r.gStyle.SetOptStat(0)
truthHist.GetYaxis().SetRangeUser(0.,8.)
truthHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
canv.Print('truthJetHist.pdf')
predHist.GetYaxis().SetRangeUser(0.,8.)
predHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
canv.Print('recoPredJetHist.pdf')
rightHist.GetYaxis().SetRangeUser(0.,1.)
rightHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
canv.Print('recoEfficiencyJetHist.pdf')

#setup 2D hists
canv = useSty.setCanvas()
truthHist = r.TH1F('truthHist','truthHist',nClasses,-0.5,nClasses-0.5)
truthHist.SetTitle('')
useSty.formatHisto(truthHist)
predHist  = r.TH1F('predHist','predHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(predHist)
predHist.SetTitle('')
rightHist = r.TH1F('rightHist','rightHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(rightHist)
rightHist.SetTitle('')
wrongHist = r.TH1F('wrongHist','wrongHist',nClasses,-0.5,nClasses-0.5)
useSty.formatHisto(wrongHist)
wrongHist.SetTitle('')

for iBin in range(1, nClasses+1):
    truthHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    predHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    rightHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])
    wrongHist.GetXaxis().SetBinLabel(iBin, binNames[iBin-1])

for true,guess,w in zip(classTestY,classPredY,classTestFW):
    truthHist.Fill(true,w)
    predHist.Fill(guess,w)
    if true==guess: rightHist.Fill(true,w)
    else: wrongHist.Fill(true,w)
        
firstBinVal = -1.
for iBin in range(1,truthHist.GetNbinsX()+1):
    if iBin==1: firstBinVal = truthHist.GetBinContent(iBin)
    ratio = float(truthHist.GetBinContent(iBin)) / firstBinVal
    print 'ratio for bin %g is %1.7f'%(iBin,ratio)
    
wrongHist.Add(rightHist)
rightHist.Divide(wrongHist)
effHist = r.TH1F
r.gStyle.SetOptStat(0)
truthHist.GetYaxis().SetRangeUser(0.,8.)
truthHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
canv.Print('truthJetHist.pdf')
predHist.GetYaxis().SetRangeUser(0.,8.)
predHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
canv.Print('predJetHist.pdf')
rightHist.GetYaxis().SetRangeUser(0.,1.)
rightHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
canv.Print('efficiencyJetHist.pdf')

#generate weights for the 2D hists    
sumwProcCatMapReco = {}
sumwProcCatMapPred = {}
sumwProcMap = {}
for iProc in range(nClasses):
    sumwProcMap[iProc] = np.sum(classTestFW*(classTestY==iProc))
    for jProc in range(nClasses):
        sumwProcCatMapPred[(iProc,jProc)] = np.sum(classTestFW*(classTestY==iProc)*(classPredY==jProc))
        sumwProcCatMapReco[(iProc,jProc)] = np.sum(classTestFW*(classTestY==iProc)*(classTestR==jProc))
sumwCatMapReco = {}
sumwCatMapPred = {}
for iProc in range(nClasses):
    sumwCatMapPred[iProc] = np.sum(classTestFW*(classPredY==iProc))
    sumwCatMapReco[iProc] = np.sum(classTestFW*(classTestR==iProc))
    #sumwCatMapPred[iProc] = np.sum(classTestFW*(classTred==iProc)*(classTestY!=0)) #don't count bkg here
    #sumwCatMapReco[iProc] = np.sum(classTestFW*(classTestR==iProc)*(classTestY!=0))

#fill the 2D hists
nBinsX=nClasses
nBinsY=nClasses
procHistReco = r.TH2F('procHistReco','procHistReco', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
procHistReco.SetTitle('')
prettyHist(procHistReco)
procHistPred = r.TH2F('procHistPred','procHistPred', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
procHistPred.SetTitle('')
prettyHist(procHistPred)
catHistReco  = r.TH2F('catHistReco','catHistReco', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
catHistReco.SetTitle('')
prettyHist(catHistReco)
catHistPred  = r.TH2F('catHistPred','catHistPred', nBinsX, -0.5, nBinsX-0.5, nBinsY, -0.5, nBinsY-0.5)
catHistPred.SetTitle('')
prettyHist(catHistPred)

for iBin in range(nClasses):
    procHistReco.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistReco.GetXaxis().SetTitle('gen bin')
    procHistReco.GetYaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistReco.GetYaxis().SetTitle('reco bin')
    procHistPred.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistPred.GetXaxis().SetTitle('gen bin')
    procHistPred.GetYaxis().SetBinLabel( iBin+1, binNames[iBin] )
    procHistPred.GetYaxis().SetTitle('reco bin')
    catHistReco.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    catHistReco.GetXaxis().SetTitle('gen bin')
    catHistReco.GetYaxis().SetBinLabel( iBin+1, binNames[iBin] )
    catHistReco.GetYaxis().SetTitle('reco bin')
    catHistPred.GetXaxis().SetBinLabel( iBin+1, binNames[iBin] )
    catHistPred.GetXaxis().SetTitle('gen bin')
    catHistPred.GetYaxis().SetBinLabel( iBin+1, binNames[iBin] )
    catHistPred.GetYaxis().SetTitle('reco bin')

for iProc in range(nClasses):
    for jProc in range(nClasses):
        procWeightReco = 100. * sumwProcCatMapReco[(iProc,jProc)] / sumwProcMap[iProc]
        procWeightPred = 100. * sumwProcCatMapPred[(iProc,jProc)] / sumwProcMap[iProc]
        catWeightReco  = 100. * sumwProcCatMapReco[(iProc,jProc)] / sumwCatMapReco[jProc]
        catWeightPred  = 100. * sumwProcCatMapPred[(iProc,jProc)] / sumwCatMapPred[jProc]

        procHistReco.Fill(iProc, jProc, procWeightReco)
        procHistPred.Fill(iProc, jProc, procWeightPred)
        catHistReco.Fill(iProc, jProc, catWeightReco)
        catHistPred.Fill(iProc, jProc, catWeightPred)

#draw the 2D hists
canv = r.TCanvas()
r.gStyle.SetPaintTextFormat('2.0f')
prettyHist(procHistReco)
procHistReco.Draw('colz,text')
canv.Print('procJetHistReco.pdf')
prettyHist(catHistReco)
catHistReco.Draw('colz,text')
canv.Print('catJetHistReco.pdf')
prettyHist(procHistPred)
procHistPred.Draw('colz,text')
canv.Print('procJetHistPred.pdf')
prettyHist(catHistPred)
catHistPred.Draw('colz,text')
canv.Print('catJetHistPred.pdf')
# get feature importances
plt.figure(1)
xg.plot_importance(jetModel)
plt.show()
#plt.savefig('classImportances.pdf')
#do some optimisation with new diphoton BDT
testingDiphoClass = xg.DMatrix(diphoTestA, label=diphoTestB, weight=diphoTestFW, feature_names=classVars)
predDiphoClass = classModel.predict(testingDiphoClass).reshape(diphoTestB.shape[0],nClasses)
predDiphoClass = np.argmax(predDiphoClass, axis=1)
