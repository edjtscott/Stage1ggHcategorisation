#usual imports
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
procFileMap = {'ggh':'powheg_ggH.root', 'vbf':'powheg_VBF.root', #'tth':'powheg_ttH.root', 'vh':'powheg_VH.root',
               'dipho':'Dipho.root', 'gjet_anyfake':'GJet.root', 'qcd_anyfake':'QCD.root'}
theProcs = procFileMap.keys()
#signals     = ['ggh','vbf','tth','vh']
signals     = ['ggh','vbf']
backgrounds = ['dipho','gjet_anyfake','qcd_anyfake']

#define the different sets of variables used
from variableDefinitions import allVarsGen, diphoVars
queryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_leadIDMVA>-0.9) and (dipho_subleadIDMVA>-0.9) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25)'

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      trainFile = upr.open('%s/%s'%(trainDir,fn)) 
      if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
      elif proc in backgrounds: trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
      else: raise Exception('Error did not recognise process %s !'%proc)
      trainFrames[proc] = trainTree.pandas.df(allVarsGen).query(queryString)
      trainFrames[proc]['proc'] = proc
  print 'got trees'
  
  #create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  sigSumW = np.sum( trainTotal[trainTotal.HTXSstage1_1_cat>0.01]['weight'].values )
  bkgSumW = np.sum( trainTotal[trainTotal.HTXSstage1_1_cat==0]['weight'].values )
  weightRatio = bkgSumW/sigSumW
  print 'Weight info:'
  print 'sigSumW %.6f'%sigSumW
  print 'bkgSumW %.6f'%bkgSumW
  print 'S/B ratio %.6f'%(1./weightRatio)

  #add extra info to dataframe
  print 'about to add extra columns'
  trainTotal['truthDipho'] = trainTotal.apply(truthDipho, axis=1)
  trainTotal['diphoWeight'] = trainTotal.apply(diphoWeight, axis=1)
  trainTotal['altDiphoWeight'] = trainTotal.apply(altDiphoWeight, axis=1, args=[weightRatio])
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

#define the indices shuffle (useful to keep this separate so it can be re-used)
theShape = trainTotal.shape[0]
diphoShuffle = np.random.permutation(theShape)
diphoTrainLimit = int(theShape*trainFrac)
diphoValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for diphoton training
diphoX  = trainTotal[diphoVars].values
diphoY  = trainTotal['truthDipho'].values
diphoTW = trainTotal['diphoWeight'].values
diphoAW = trainTotal['altDiphoWeight'].values
diphoFW = trainTotal['weight'].values
diphoM  = trainTotal['dipho_mass'].values
del trainTotal

diphoX  = diphoX[diphoShuffle]
diphoY  = diphoY[diphoShuffle]
diphoTW = diphoTW[diphoShuffle]
diphoAW = diphoAW[diphoShuffle]
diphoFW = diphoFW[diphoShuffle]
diphoM  = diphoM[diphoShuffle]

diphoTrainX,  diphoValidX,  diphoTestX  = np.split( diphoX,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainY,  diphoValidY,  diphoTestY  = np.split( diphoY,  [diphoTrainLimit,diphoValidLimit] )
diphoTrainTW, diphoValidTW, diphoTestTW = np.split( diphoTW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainAW, diphoValidAW, diphoTestAW = np.split( diphoAW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainFW, diphoValidFW, diphoTestFW = np.split( diphoFW, [diphoTrainLimit,diphoValidLimit] )
diphoTrainM,  diphoValidM,  diphoTestM  = np.split( diphoM,  [diphoTrainLimit,diphoValidLimit] )

#build the background discrimination BDT
trainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainTW, feature_names=diphoVars)
testingDipho  = xg.DMatrix(diphoTestX,  label=diphoTestY,  weight=diphoTestFW,  feature_names=diphoVars)
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
print 'about to train diphoton BDT'
diphoModel = xg.train(trainParams, trainingDipho)
print 'done'

#save it
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
diphoModel.save_model('%s/diphoModel%s.model'%(modelDir,paramExt))
print 'saved as %s/diphoModel%s.model'%(modelDir,paramExt)

#build same thing but with equalised weights
altTrainingDipho = xg.DMatrix(diphoTrainX, label=diphoTrainY, weight=diphoTrainAW, feature_names=diphoVars)
print 'about to train alternative diphoton BDT'
altDiphoModel = xg.train(trainParams, altTrainingDipho)
print 'done'

#save it
altDiphoModel.save_model('%s/altDiphoModel%s.model'%(modelDir,paramExt))
print 'saved as %s/altDiphoModel%s.model'%(modelDir,paramExt)

#check performance of each training
diphoPredYxcheck = diphoModel.predict(trainingDipho)
diphoPredY = diphoModel.predict(testingDipho)
print 'Default training performance:'
print 'area under roc curve for training set = %1.3f'%( roc_auc_score(diphoTrainY, diphoPredYxcheck, sample_weight=diphoTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( roc_auc_score(diphoTestY, diphoPredY, sample_weight=diphoTestFW) )

altDiphoPredYxcheck = altDiphoModel.predict(trainingDipho)
altDiphoPredY = altDiphoModel.predict(testingDipho)
print 'Alternative training performance:'
print 'area under roc curve for training set = %1.3f'%( roc_auc_score(diphoTrainY, altDiphoPredYxcheck, sample_weight=diphoTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( roc_auc_score(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW) )

exit("Skip plotting for now") #FIXME maybe make configurable?

#make some plots 
plotDir = trainDir.replace('trees','plots')
plotDir = '%s/%s'%paramExt
if not path.isdir(plotDir): 
  system('mkdir -p %s'%plotDir)
bkgEff, sigEff, nada = roc_curve(diphoTestY, diphoPredY, sample_weight=diphoTestFW)
plt.figure(1)
plt.plot(bkgEff, sigEff)
plt.xlabel('Background efficiency')
plt.ylabel('Signal efficiency')
#plt.show()
plt.savefig('%s/diphoROC.pdf'%plotDir)
bkgEff, sigEff, nada = roc_curve(diphoTestY, altDiphoPredY, sample_weight=diphoTestFW)
plt.figure(2)
plt.plot(bkgEff, sigEff)
plt.xlabel('Background efficiency')
plt.ylabel('Signal efficiency')
#plt.show()
plt.savefig('%s/altDiphoROC.pdf'%plotDir)
plt.figure(3)
xg.plot_importance(diphoModel)
#plt.show()
plt.savefig('%s/diphoImportances.pdf'%plotDir)
plt.figure(4)
xg.plot_importance(altDiphoModel)
#plt.show()
plt.savefig('%s/altDiphoImportances.pdf'%plotDir)

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

sigScoreHist.SetLineColor(r.kBlue)
sigScoreHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
theCanv.SaveAs('%s/sigScore.pdf'%plotDir)
bkgScoreHist.SetLineColor(r.kRed)
bkgScoreHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
theCanv.SaveAs('%s/bkgScore.pdf'%plotDir)

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
theCanv.SaveAs('%s/outputScores.pdf'%plotDir)

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

sigScoreHist.SetLineColor(r.kBlue)
sigScoreHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
theCanv.SaveAs('%s/altSigScore.pdf'%plotDir)
bkgScoreHist.SetLineColor(r.kRed)
bkgScoreHist.Draw('hist')
useSty.drawCMS()
useSty.drawEnPu(lumi='35.9 fb^{-1}')
theCanv.SaveAs('%s/altBkgScore.pdf'%plotDir)

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
useSty.drawEnPu(lumi='%2.1f fb^{-1}'%opts.intLumi)
theCanv.SaveAs('%s/altOutputScores.pdf'%plotDir)
