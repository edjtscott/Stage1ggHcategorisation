#usual imports
import numpy as np
import xgboost as xg
import pickle
from addRowFunctions import addPt, truthDipho, truthClass, reco, fullWeight, normWeight, diphoWeight, altDiphoWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
import pandas as pd
import ROOT as r
from root_numpy import tree2array, testdata, list_branches, fill_hist
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import usefulStyle as useSty

#setup global variables
trainFrac = 0.7
validFrac = 0.1
nClasses = 9

#get trees from files, put them in data frames
#procFileMap = {'ggh':'ggH.root', 'dipho':'Dipho.root', 'gjet':'GJet.root', 'qcd':'QCD.root', 'Data':'Data.root' }
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'tth':'ttH.root', 'wzh':'VH.root', 'dipho':'Dipho.root', 'gjet':'GJet.root', 'qcd':'QCD.root'}
theProcs = procFileMap.keys()

trainDir = '/vols/cms/es811/Stage1categorisation/2016/trees'
trainFrames = {}
for proc,fn in procFileMap.iteritems():
    trainFile   = r.TFile('%s/%s'%(trainDir,fn))
    if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
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

#then filter out the events into only those with the phase space we are interested in
trainTotal = trainTotal[trainTotal.CMS_hgg_mass>100.]
print 'done lower mass cut'
trainTotal = trainTotal[trainTotal.CMS_hgg_mass<180.]
print 'done upper mass cut'

#some extra cuts that are applied for diphoton BDT in the AN
trainTotal = trainTotal[trainTotal.leadmva>-0.9]
print 'done leadIDMVA cut'
trainTotal = trainTotal[trainTotal.subleadmva>-0.9]
print 'done subleadIDMVA cut'
trainTotal = trainTotal[trainTotal.leadptom>0.333]
print 'done lead pT/m cut'
trainTotal = trainTotal[trainTotal.subleadptom>0.25]
print 'done sublead pT/m cut'

#TODO: add some basic checks of signal counts here
#print 35.9 * np.sum( trainTotal[trainTotal.proc=='ggh']['weight'].values )
trainTotal.to_pickle('%s/trainTotal.pkl'%trainDir)
print 'frame saved as %s/trainTotal.pkl'%trainDir
exit('Completed the construction of the dataframe!!!')

#trainTotal
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
print 'all columns added'

#try alternative weight scenario just to check
trainTotal['altDiphoWeight'] = trainTotal.apply(altDiphoWeight, axis=1)

#all necessary variables added - good idea to save at this point! 
trainTotal.to_pickle('trainTotal.pkl')
print 'frame saved as trainTotal.pkl'

#read in dataframe if above steps done before
trainTotal = pd.read_pickle('trainTotal.pkl')

#now remove data for training
trainTotal = trainTotal[trainTotal.stage1cat>-1.]
print 'done first stage 1 cut'
trainTotal = trainTotal[trainTotal.stage1cat<12.]
print 'done second stage 1 cut'
trainTotal = trainTotal[trainTotal.stage1cat!=1]
print 'done third stage 1 cut'
trainTotal = trainTotal[trainTotal.stage1cat!=2]
print 'done fourth and final stage 1 cut'

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

#build the background discrimination BDT
trainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainTW, feature_names=diphoVars)
testingDipho  = xg.DMatrix(diphoTestX,  label=diphoTestY,  weight=diphoTestFW,  feature_names=diphoVars)
paramDipho = {}
paramDipho['objective'] = 'binary:logistic'
#paramDipho['max_delta_step'] = 5
print 'about to train diphoton BDT'
diphoModel = xg.train(paramDipho, trainingDipho)
print 'done'

#save it
diphoModel.save_model('diphoModel.model')
print 'saved as diphoModel.model'
pickle.dump(diphoModel, open('diphoModel.pickle.dat', 'wb'))
print 'and diphoModel.pickle.dat'
diphoAW  = trainTotal['altDiphoWeight'].values
diphoAW  = diphoAW[diphoShuffle]
diphoTrainAW,  diphoValidAW,  diphoTestAW  = np.split( diphoAW,  [diphoTrainLimit,diphoValidLimit] )
altTrainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainAW, feature_names=diphoVars)
altParamDipho = {}
altParamDipho['objective'] = 'binary:logistic'
#paramDipho['max_delta_step'] = 5
print 'about to train diphoton BDT'
altDiphoModel = xg.train(altParamDipho, altTrainingDipho)
print 'done'

#save it
altDiphoModel.save_model('altDiphoModel.model')
print 'saved as altDiphoModel.model'
pickle.dump(altDiphoModel, open('altDiphoModel.pickle.dat', 'wb'))
print 'and altDiphoModel.pickle.dat'

#optional loading of premade models and dataframes
#trainTotal.to_pickle('trainTotal.pkl')
#trainTotal = pd.read_pickle('trainTotal.pkl')
#classTotal = trainTotal[trainTotal.proc=='ggh']
#diphoModel = xg.Booster()
#diphoModel.load_model('diphoModel.model')
#altDiphoModel = xg.Booster()
#altDiphoModel.load_model('altDiphoModel.model')
#classModel = xg.Booster()
#classModel.load_model('classModel.model')
#altClassModel = xg.Booster()
#altClassModel.load_model('altClassModel.model')
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
#from matplotlib import rc
#rc('text', usetex=True)
bkgEff, sigEff, nada = roc_curve(diphoTestY, diphoPredY, sample_weight=diphoTestFW)
plt.figure(1)
plt.plot(bkgEff, sigEff)
plt.xlabel('Background efficiency')
plt.ylabel('Signal efficiency')
#plt.show()
plt.savefig('diphoROC.pdf')
bkgEff, sigEff, nada = roc_curve(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW)
plt.figure(2)
plt.plot(bkgEff, sigEff)
plt.xlabel('Background efficiency')
plt.ylabel('Signal efficiency')
#plt.show()
plt.savefig('altDiphoROC.pdf')
plt.figure(3)
xg.plot_importance(diphoModel)
#plt.show()
plt.savefig('diphoImportances.pdf')
plt.figure(4)
xg.plot_importance(altDiphoModel)
#plt.show()
plt.savefig('altDiphoImportances.pdf')

#draw sig vs background distribution
nOutputBins = 50
theCanv = useSty.setCanvas()
sigScoreW = diphoTestFW * (diphoTestY==1)
sigScoreHist = r.TH1F('sigScoreHist', 'sigScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(sigScoreHist)
sigScoreHist.SetTitle('')
sigScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(sigScoreHist, diphoPredY, weights=sigScoreW)
bkgScoreW = diphoTestFW * (diphoTestY==0)
bkgScoreHist = r.TH1F('bkgScoreHist', 'bkgScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(bkgScoreHist)
bkgScoreHist.SetTitle('')
bkgScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(bkgScoreHist, diphoPredY, weights=bkgScoreW)


#apply transformation to flatten ggH
for iBin in range(1,nOutputBins+1):
    sigVal = sigScoreHist.GetBinContent(iBin)
    bkgVal = bkgScoreHist.GetBinContent(iBin)
    sigScoreHist.SetBinContent(iBin, 1.)
    if sigVal > 0.: 
        bkgScoreHist.SetBinContent(iBin, bkgVal/sigVal)
    else:
        bkgScoreHist.SetBinContent(iBin, 0)
        
sigScoreHist.Scale(1./sigScoreHist.Integral())
bkgScoreHist.Scale(1./bkgScoreHist.Integral())
sigScoreHist.SetLineColor(r.kBlue)
sigScoreHist.Draw('hist')
bkgScoreHist.SetLineColor(r.kRed)
bkgScoreHist.Draw('hist,same')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
theCanv.SaveAs('outputScores.pdf')

#draw sig vs background distribution
theCanv = useSty.setCanvas()
sigScoreW = diphoTestFW * (diphoTestY==1)
sigScoreHist = r.TH1F('sigScoreHist', 'sigScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(sigScoreHist)
sigScoreHist.SetTitle('')
sigScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(sigScoreHist, altDiphoPredY, weights=sigScoreW)
bkgScoreW = diphoTestFW * (diphoTestY==0)
bkgScoreHist = r.TH1F('bkgScoreHist', 'bkgScoreHist', nOutputBins, 0., 1.)
useSty.formatHisto(bkgScoreHist)
bkgScoreHist.SetTitle('')
bkgScoreHist.GetXaxis().SetTitle('Diphoton BDT score')
fill_hist(bkgScoreHist, altDiphoPredY, weights=bkgScoreW)

for iBin in range(1,nOutputBins+1):
    sigVal = sigScoreHist.GetBinContent(iBin)
    bkgVal = bkgScoreHist.GetBinContent(iBin)
    sigScoreHist.SetBinContent(iBin, 1.)
    if sigVal > 0.: 
        bkgScoreHist.SetBinContent(iBin, bkgVal/sigVal)
    else:
        bkgScoreHist.SetBinContent(iBin, 0)
        
sigScoreHist.Scale(1./sigScoreHist.Integral())
bkgScoreHist.Scale(1./bkgScoreHist.Integral())
sigScoreHist.SetLineColor(r.kBlue)
sigScoreHist.Draw('hist')
bkgScoreHist.SetLineColor(r.kRed)
bkgScoreHist.Draw('hist,same')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
theCanv.SaveAs('altOutputScores.pdf')

#now estimate significance using the amount of background in a plus/minus 1 sigma window
#reco categorisation version
lumi = 35.9 / (1. - (trainFrac+validFrac))
for iProc in range(nClasses):
    bestCut = -.1
    bestSignif = -1.
    bestS = -1.
    bestB = -1.
    for cut in np.arange(0.0,0.1,0.001):
        sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
        sigWeights = diphoTestFW * (diphoTestY==1) * (diphoTestB==iProc) * (diphoPredY>cut) * (diphoTestR==iProc)
        fill_hist(sigHist, diphoTestM, weights=sigWeights)
        sigCount = 0.68 * lumi * sigHist.Integral() 
        sigWidth = getRealSigma(sigHist)
        #print 'sigwidth is %1.3f'%sigWidth
        bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
        bkgWeights = diphoTestFW * (diphoTestY==0) * (diphoPredY>cut) * (diphoTestR==iProc)
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

#now estimate two-class significance
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
            sigWeightsHi = diphoTestFW * (diphoTestY==1) * (diphoTestB==iProc) * (diphoPredY>cutHi) * (diphoTestR==iProc)
            fill_hist(sigHistHi, diphoTestM, weights=sigWeightsHi)
            sigCountHi = 0.68 * lumi * sigHistHi.Integral() 
            sigWidthHi = getRealSigma(sigHistHi)
            #print 'sigwidth is %1.3f'%sigWidth
            bkgHistHi = r.TH1F('bkgHistHiTemp','bkgHistHiTemp',160,100,180)
            bkgWeightsHi = diphoTestFW * (diphoTestY==0) * (diphoPredY>cutHi) * (diphoTestR==iProc)
            fill_hist(bkgHistHi, diphoTestM, weights=bkgWeightsHi)
            bkgCountHi = lumi * computeBkg(bkgHistHi, sigWidthHi)
            theSignifHi = getAMS(sigCountHi, bkgCountHi)
            sigHistLo = r.TH1F('sigHistLoTemp','sigHistLoTemp',160,100,180)
            sigWeightsLo = diphoTestFW * (diphoTestY==1) * (diphoTestB==iProc) * (diphoPredY<cutHi) * (diphoPredY>cutLo) * (diphoTestR==iProc)
            fill_hist(sigHistLo, diphoTestM, weights=sigWeightsLo)
            sigCountLo = 0.68 * lumi * sigHistLo.Integral() 
            sigWidthLo = getRealSigma(sigHistLo)
            #print 'sigwidth is %1.3f'%sigWidth
            bkgHistLo = r.TH1F('bkgHistLoTemp','bkgHistLoTemp',160,100,180)
            bkgWeightsLo = diphoTestFW * (diphoTestY==0) * (diphoPredY<cutHi) * (diphoPredY>cutLo) * (diphoTestR==iProc)
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
    print 'for proc %g the best outcome was tot signif %1.2f:'%(iProc,bestSignif)
    print 'cutLo %1.3f, Slo %1.2f, Blo %1.2f, signifLo %1.2f'%(bestCutLo,bestSlo,bestBlo,bestSignifLo)
    print 'cutHi %1.3f, Shi %1.2f, Bhi %1.2f, signifHi %1.2f'%(bestCutHi,bestShi,bestBhi,bestSignifHi)
    print

#now estimate significance using the amount of background in a plus/minus 1 sigma window
#reco categorisation version
lumi = 35.9 / (1. - (trainFrac+validFrac))
for iProc in range(nClasses):
    bestCut = -.1
    bestSignif = -1.
    bestS = -1.
    bestB = -1.
    for cut in np.arange(0.7,1.,0.0025):
        sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
        sigWeights = diphoTestFW * (diphoTestY==1) * (diphoTestB==iProc) * (altDiphoPredY>cut) * (diphoTestR==iProc)
        fill_hist(sigHist, diphoTestM, weights=sigWeights)
        sigCount = 0.68 * lumi * sigHist.Integral() 
        sigWidth = getRealSigma(sigHist)
        #print 'sigwidth is %1.3f'%sigWidth
        bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
        bkgWeights = diphoTestFW * (diphoTestY==0) * (altDiphoPredY>cut) * (diphoTestR==iProc)
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

#now estimate two-class significance
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
            sigWeightsHi = diphoTestFW * (diphoTestY==1) * (diphoTestB==iProc) * (altDiphoPredY>cutHi) * (diphoTestR==iProc)
            fill_hist(sigHistHi, diphoTestM, weights=sigWeightsHi)
            sigCountHi = 0.68 * lumi * sigHistHi.Integral() 
            sigWidthHi = getRealSigma(sigHistHi)
            #print 'sigwidth is %1.3f'%sigWidth
            bkgHistHi = r.TH1F('bkgHistHiTemp','bkgHistHiTemp',160,100,180)
            bkgWeightsHi = diphoTestFW * (diphoTestY==0) * (altDiphoPredY>cutHi) * (diphoTestR==iProc)
            fill_hist(bkgHistHi, diphoTestM, weights=bkgWeightsHi)
            bkgCountHi = lumi * computeBkg(bkgHistHi, sigWidthHi)
            theSignifHi = getAMS(sigCountHi, bkgCountHi)
            sigHistLo = r.TH1F('sigHistLoTemp','sigHistLoTemp',160,100,180)
            sigWeightsLo = diphoTestFW * (diphoTestY==1) * (diphoTestB==iProc) * (altDiphoPredY<cutHi) * (altDiphoPredY>cutLo) * (diphoTestR==iProc)
            fill_hist(sigHistLo, diphoTestM, weights=sigWeightsLo)
            sigCountLo = 0.68 * lumi * sigHistLo.Integral() 
            sigWidthLo = getRealSigma(sigHistLo)
            #print 'sigwidth is %1.3f'%sigWidth
            bkgHistLo = r.TH1F('bkgHistLoTemp','bkgHistLoTemp',160,100,180)
            bkgWeightsLo = diphoTestFW * (diphoTestY==0) * (altDiphoPredY<cutHi) * (altDiphoPredY>cutLo) * (diphoTestR==iProc)
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
    print 'for proc %g the best outcome was tot signif %1.2f:'%(iProc,bestSignif)
    print 'cutLo %1.3f, Slo %1.2f, Blo %1.2f, signifLo %1.2f'%(bestCutLo,bestSlo,bestBlo,bestSignifLo)
    print 'cutHi %1.3f, Shi %1.2f, Bhi %1.2f, signifHi %1.2f'%(bestCutHi,bestShi,bestBhi,bestSignifHi)
    print
