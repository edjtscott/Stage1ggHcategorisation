#usual imports
import numpy as np
import xgboost as xg
import pickle
from addRowFunctions import addPt, truthDipho, truthClass, reco, fullWeight, normWeight, diphoWeight, truthJets, jetWeight
from otherHelpers import prettyHist
import pandas as pd
import ROOT as r
from root_numpy import tree2array, testdata, list_branches, fill_hist
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import usefulStyle as useSty

#setup global variables
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
              #'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_LeadJPt','dijet_SubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta']
jetVars  = ['n_rec_jets','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              #'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_LeadJPt','dijet_SubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta']

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root', 'dipho':'Dipho.root', 'gjet':'GJet.root', 'qcd':'QCD.root' }
theProcs = procFileMap.keys()

trainDir = '../trainTrees'
trainFrames = {}
for proc,fn in procFileMap.iteritems():
    trainFile   = r.TFile('%s/%s'%(trainDir,fn))
    if 'ggh' in proc or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
    else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
    trainTree.SetBranchStatus('nvtx',0)
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
print 'created total frame'

#then filter out the events into only those with the phase space we are interested in
trainTotal = trainTotal[trainTotal.CMS_hgg_mass>100.]
print 'done lower mass cut'
trainTotal = trainTotal[trainTotal.CMS_hgg_mass<180.]
print 'done upper mass cut'
trainTotal = trainTotal[trainTotal.stage1cat>-1.]
print 'done first stage 1 cut'
trainTotal = trainTotal[trainTotal.stage1cat<12.]
print 'done second stage 1 cut'
trainTotal = trainTotal[trainTotal.stage1cat!=1]
print 'done third stage 1 cut'
trainTotal = trainTotal[trainTotal.stage1cat!=2]
print 'done fourth and final stage 1 cut'

#some extra cuts that are applied for diphoton BDT in the AN
trainTotal = trainTotal[trainTotal.leadmva>-0.9]
print 'done leadIDMVA cut'
trainTotal = trainTotal[trainTotal.subleadmva>-0.9]
print 'done subleadIDMVA cut'
trainTotal = trainTotal[trainTotal.leadptom>0.33]
print 'done lead pT/m cut'
trainTotal = trainTotal[trainTotal.subleadptom>0.25]
print 'done sublead pT/m cut'
print 'now add the various columns'

#add diphoton pt as a column
trainTotal['diphopt'] = trainTotal.apply(addPt, axis=1)
#add column corresponding to truth bin
trainTotal['truthDipho'] = trainTotal.apply(truthDipho,axis=1)
#add column corresponding to truth bin
trainTotal['truthClass'] = trainTotal.apply(truthClass,axis=1)
#add column corresponding to reco bin
trainTotal['reco'] = trainTotal.apply(reco,axis=1)
#add column with full weights for testing
trainTotal['fullWeight'] = trainTotal.apply(fullWeight,axis=1)
#add column with *normalised weights for multiclass training only*
trainTotal['normedWeight'] = trainTotal.apply(normWeight,axis=1)
#add column with weights for dipho BDT
trainTotal['diphoWeight'] = trainTotal.apply(diphoWeight,axis=1)
#add column with truth number of jets
trainTotal['truthJets'] = trainTotal.apply(truthJets,axis=1)
#add column with truth number of jets
trainTotal['jetWeight'] = trainTotal.apply(jetWeight,axis=1)
print 'all columns added'

#all necessary variables added - good idea to save at this point! 
trainTotal.to_pickle('trainTotal.pkl')
print 'frame saved as trainTotal.pkl'

#read in dataframe if above steps done before
trainTotal = pd.read_pickle('trainTotal.pkl')

#add column with truth number of jets
#FIXME to be removed
trainTotal['truthJets'] = trainTotal.apply(truthJets,axis=1)
#add column with weight for jet training
#FIXME to be removed
trainTotal['jetWeight'] = trainTotal.apply(jetWeight,axis=1)
trainTotal.to_pickle('trainTotalTemp.pkl')

#define the frame for the class training
classTotal = trainTotal[trainTotal.proc=='ggh']
classTotal = classTotal[classTotal.stage1cat>-1.]
print 'done first stage 1 cut'
classTotal = classTotal[classTotal.stage1cat<12.]
print 'done second stage 1 cut'
classTotal = classTotal[classTotal.stage1cat!=1]
print 'done third stage 1 cut'
classTotal = classTotal[classTotal.stage1cat!=2]
print 'done fourth and final stage 1 cut'
classTotal[classTotal.truthJets<0]

#define the indices shuffle (useful to keep this separate so it can be re-used)
theShape = trainTotal.shape[0]
diphoShuffle = np.random.permutation(theShape)
diphoTrainLimit = int(theShape*trainFrac)
diphoValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for diphoton training
diphoX  = trainTotal[diphoVars].values #diphoton BDT input variables
diphoY  = trainTotal['truthDipho'].values #truth/target values (i.e. signal or background)
diphoTW = trainTotal['diphoWeight'].values #weights for training
diphoFW = trainTotal['fullWeight'].values #weights corresponding to number of events
diphoR  = trainTotal['reco'].values
diphoM  = trainTotal['CMS_hgg_mass'].values
diphoA  = trainTotal[classVars].values #multi-class inputs
diphoB  = trainTotal['truthClass'].values #truth stage 1 values

diphoX  = diphoX[diphoShuffle]
diphoY  = diphoY[diphoShuffle]
diphoTW = diphoTW[diphoShuffle]
diphoFW = diphoFW[diphoShuffle]
diphoR  = diphoR[diphoShuffle]
diphoM  = diphoM[diphoShuffle]
diphoA  = diphoA[diphoShuffle]
diphoB  = diphoB[diphoShuffle]

diphoTrainX,  diphoValidX,  diphoTestX  = np.split( diphoX,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainY,  diphoValidY,  diphoTestY  = np.split( diphoY,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainTW, diphoValidTW, diphoTestTW = np.split( diphoTW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainFW, diphoValidFW, diphoTestFW = np.split( diphoFW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainR,  diphoValidR,  diphoTestR  = np.split( diphoR,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainM,  diphoValidM,  diphoTestM  = np.split( diphoM,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainA,  diphoValidA,  diphoTestA  = np.split( diphoA,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainB,  diphoValidB,  diphoTestB  = np.split( diphoB,  [diphoTrainLimit,diphoValidLimit] )

#build the dipho matrices
trainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainTW, feature_names=diphoVars)
testingDipho  = xg.DMatrix(diphoTestX,  label=diphoTestY,  weight=diphoTestFW,  feature_names=diphoVars)
#and the alternate ones
diphoAW  = trainTotal['altDiphoWeight'].values
diphoAW  = diphoAW[diphoShuffle]
diphoTrainAW,  diphoValidAW,  diphoTestAW  = np.split( diphoAW,  [diphoTrainLimit,diphoValidLimit] )
altTrainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainAW, feature_names=diphoVars)

#get diphoton models
diphoModel = xg.Booster()
diphoModel.load_model('diphoModel.model')
altDiphoModel = xg.Booster()
altDiphoModel.load_model('altDiphoModel.model')
#get predicted values
diphoPredY = diphoModel.predict(testingDipho)
diphoPredYxcheck = diphoModel.predict(trainingDipho)
print diphoTestY
print diphoPredY
print roc_auc_score(diphoTrainY, diphoPredYxcheck, sample_weight=diphoTrainFW)
print roc_auc_score(diphoTestY, diphoPredY, sample_weight=diphoTestFW)

altDiphoPredY = altDiphoModel.predict(testingDipho)
print diphoTestY
print altDiphoPredY
print roc_auc_score(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW)

#shape and suffle definitions - useful to preserve here
theShape = classTotal.shape[0]
classShuffle = np.random.permutation(theShape)
classTrainLimit = int(theShape*trainFrac)
classValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for multiclass training
classX  = classTotal[classVars].values #multi-classifier input variables
classY  = classTotal['truthClass'].values #truth/target values (i.e. gen-level Stage 1 bin)
classTW = classTotal['normedWeight'].values #normalised weights for training - can be played with
classFW = classTotal['fullWeight'].values
classR  = classTotal['reco'].values
classM  = classTotal['CMS_hgg_mass'].values
classI  = classTotal[jetVars].values
classJ  = classTotal['truthJets'].values
classP  = classTotal['diphopt'].values
classJW  = classTotal['jetWeight'].values

classX  = classX[classShuffle]
classY  = classY[classShuffle]
classTW = classTW[classShuffle]
classFW = classFW[classShuffle]
classR  = classR[classShuffle]
classM  = classM[classShuffle]
classI  = classI[classShuffle]
classJ  = classJ[classShuffle]
classP  = classP[classShuffle]
classJW = classJW[classShuffle]

classTrainX,  classValidX,  classTestX  = np.split( classX,  [classTrainLimit,classValidLimit] )
classTrainY,  classValidY,  classTestY  = np.split( classY,  [classTrainLimit,classValidLimit] )
classTrainTW, classValidTW, classTestTW = np.split( classTW, [classTrainLimit,classValidLimit] )
classTrainFW, classValidFW, classTestFW = np.split( classFW, [classTrainLimit,classValidLimit] )
classTrainR,  classValidR,  classTestR  = np.split( classR,  [classTrainLimit,classValidLimit] )
classTrainM,  classValidM,  classTestM  = np.split( classM,  [classTrainLimit,classValidLimit] )
classTrainI,  classValidI,  classTestI  = np.split( classI,  [classTrainLimit,classValidLimit] )
classTrainJ,  classValidJ,  classTestJ  = np.split( classJ,  [classTrainLimit,classValidLimit] )
classTrainP,  classValidP,  classTestP  = np.split( classP,  [classTrainLimit,classValidLimit] )
classTrainJW,  classValidJW,  classTestJW  = np.split( classJW,  [classTrainLimit,classValidLimit] )

#build the jet-classifier
trainingJet = xg.DMatrix(classTrainI, label=classTrainJ, weight=classTrainJW, feature_names=jetVars)
testingJet  = xg.DMatrix(classTestI,  label=classTestJ,  weight=classTestFW,  feature_names=jetVars)
paramJet = {}
paramJet['objective'] = 'multi:softprob'
paramJet['num_class'] = nJetClasses
paramJet['max_depth'] = 10
#paramJet['early_stopping_rounds']=5
nTrees = 50
print 'about to train multi-class BDT'
jetModel = xg.train(paramJet, trainingJet, num_boost_round=nTrees)
print 'done'
#save
jetModel.save_model('jetModel_depth10ntrees50.model')
print 'saved as classModel.model'

#load it 
jetModel = xg.Booster()
jetModel.load_model('jetModel.model')

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

#couple more imports
from otherHelpers import getAMS, computeBkg, getRealSigma 
#now estimate significance using the amount of background in a ±1 sigma window
lumi = 35.9
lumi *= 1./(1. - (trainFrac+validFrac))
for iProc in range(nClasses):
    bestCut = -.1
    bestSignif = -1.
    bestS = -1.
    bestB = -1.
    for cut in np.arange(0.7,1.,0.0025):
        sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
        sigWeights = diphoTestFW * (diphoTestY==1) * (diphoTestB==iProc) * (altDiphoPredY>cut) * (predDiphoClass==iProc)
        fill_hist(sigHist, diphoTestM, weights=sigWeights)
        sigCount = 0.68 * lumi * sigHist.Integral() 
        sigWidth = getRealSigma(sigHist)
        #print 'sigwidth is %1.3f'%sigWidth
        bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
        bkgWeights = diphoTestFW * (diphoTestY==0) * (altDiphoPredY>cut) * (predDiphoClass==iProc)
        fill_hist(bkgHist, diphoTestM, weights=bkgWeights)
        bkgCount = lumi * computeBkg(bkgHist, sigWidth)
        theSignif = getAMS(sigCount, bkgCount)
        if theSignif > bestSignif: 
            bestCut = cut
            bestSignif = theSignif
            bestS = sigCount
            bestB = bkgCount
    print 'for proc %g the best outcome was:'%iProc
    print 'cut %1.3f, S %1.2f, B %1.2f, signif %1.2f'%(bestCut,bestS,bestB,bestSignif)
    print

#now estimate significance using the amount of background in a ±1 sigma window
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
    for cutLo in np.arange(0.7,1.,0.02):
        for cutHi in np.arange(cutLo,1.,0.02):
            sigHistHi = r.TH1F('sigHistHiTemp','sigHistHiTemp',160,100,180)
            sigWeightsHi = diphoTestFW * (diphoTestY==1) * (diphoTestB==iProc) * (altDiphoPredY>cutHi) * (predDiphoClass==iProc)
            fill_hist(sigHistHi, diphoTestM, weights=sigWeightsHi)
            sigCountHi = 0.68 * lumi * sigHistHi.Integral() 
            sigWidthHi = getRealSigma(sigHistHi)
            #print 'sigwidth is %1.3f'%sigWidth
            bkgHistHi = r.TH1F('bkgHistHiTemp','bkgHistHiTemp',160,100,180)
            bkgWeightsHi = diphoTestFW * (diphoTestY==0) * (altDiphoPredY>cutHi) * (predDiphoClass==iProc)
            fill_hist(bkgHistHi, diphoTestM, weights=bkgWeightsHi)
            bkgCountHi = lumi * computeBkg(bkgHistHi, sigWidthHi)
            theSignifHi = getAMS(sigCountHi, bkgCountHi)
            sigHistLo = r.TH1F('sigHistLoTemp','sigHistLoTemp',160,100,180)
            sigWeightsLo = diphoTestFW * (diphoTestY==1) * (diphoTestB==iProc) * (altDiphoPredY<cutHi) * (altDiphoPredY>cutLo) * (predDiphoClass==iProc)
            fill_hist(sigHistLo, diphoTestM, weights=sigWeightsLo)
            sigCountLo = 0.68 * lumi * sigHistLo.Integral() 
            sigWidthLo = getRealSigma(sigHistLo)
            #print 'sigwidth is %1.3f'%sigWidth
            bkgHistLo = r.TH1F('bkgHistLoTemp','bkgHistLoTemp',160,100,180)
            bkgWeightsLo = diphoTestFW * (diphoTestY==0) * (altDiphoPredY<cutHi) * (altDiphoPredY>cutLo) * (predDiphoClass==iProc)
            fill_hist(bkgHistLo, diphoTestM, weights=bkgWeightsLo)
            bkgCountLo = lumi * computeBkg(bkgHistLo, sigWidthLo)
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
    print 'for proc %g the best outcome was tot signif %1.3f:'%(iProc,bestSignif)
    print 'cutLo %1.3f, Slo %1.2f, Blo %1.2f, signifLo %1.2f'%(bestCutLo,bestSlo,bestBlo,bestSignifLo)
    print 'cutHi %1.3f, Shi %1.2f, Bhi %1.2f, signifHi %1.2f'%(bestCutHi,bestShi,bestBhi,bestSignifHi)
    print
