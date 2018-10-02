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
parser.add_option('-d','--dataFrame', help='Name of data dataframe')
parser.add_option('-s','--signalFrame', help='Name of signal dataframe')
parser.add_option('-m','--modelName', help='Name of model for testing')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
if not path.isdir(frameDir): system('mkdir -p %s'%frameDir)
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir): system('mkdir -p %s'%modelDir)
plotDir  = trainDir.replace('trees','plots')
if not path.isdir(plotDir): system('mkdir -p %s'%plotDir)
plotExt = opts.modelName.replace('.model','')

#define the different sets of variables used
diphoVars  = ['leadmva','subleadmva','leadptom','subleadptom',
              'leadeta','subleadeta',
              'CosPhi','vtxprob','sigmarv','sigmawv']

#get mc and data
dataTotal  = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.signalFrame))

#choose only data sidebands
trainLow  = trainTotal[trainTotal.CMS_hgg_mass<115.]
trainHigh = trainTotal[trainTotal.CMS_hgg_mass>135.]
trainTotal = pd.concat([trainLow,trainHigh])
dataLow  = dataTotal[dataTotal.CMS_hgg_mass<115.]
dataHigh = dataTotal[dataTotal.CMS_hgg_mass>135.]
dataTotal = pd.concat([dataLow,dataHigh])

#setup the various datasets
diphoX  = trainTotal[diphoVars].values
diphoFW = trainTotal['weight'].values
diphoP  = trainTotal['proc'].values
dataX   = dataTotal[diphoVars].values
dataFW  = np.ones(dataX.shape[0])

#load the model and get the scores
diphoModel = xg.Booster()
diphoModel.load_model('%s/%s'%(modelDir,opts.modelName))
diphoMatrix = xg.DMatrix(diphoX, label=diphoFW, weight=diphoFW, feature_names=diphoVars)
dataMatrix  = xg.DMatrix(dataX,  label=dataFW,  weight=dataFW,  feature_names=diphoVars)
diphoScores = diphoModel.predict(diphoMatrix)
dataScores  = diphoModel.predict(dataMatrix)

#make hist
r.gStyle.SetOptStat(0)
r.gROOT.SetBatch(True)
dataHist = r.TH1F('dataCheckHist','dataCheckHist',50,0.,1.)
fill_hist(dataHist, dataScores, dataFW)
print 'Integral of data hist is %1.3f'%dataHist.Integral()
dataHist.Scale(1./dataHist.Integral())
dataHist.SetMarkerStyle(r.kDot)
dataHist.SetMarkerColor(r.kBlack)
dataHist.SetMarkerSize(3)
dataHist.SetLineColor(r.kBlack)
diphoHist = r.TH1F('diphoCheckHist','diphoCheckHist',50,0.,1.)
fill_hist(diphoHist, diphoScores, diphoFW)
print 'Integral of dipho hist is %1.3f'%diphoHist.Integral()
diphoHist.Scale(1./diphoHist.Integral())
diphoHist.SetLineColor(r.kGreen+1)
canvA = r.TCanvas()
diphoHist.Draw('hist')
dataHist.Draw('P,same')
canvA.Print('%s/dataMCdiphoBDT_full_%s.png'%(plotDir,plotExt))
canvA.Print('%s/dataMCdiphoBDT_full_%s.pdf'%(plotDir,plotExt))

#make ratio
ratioPlot = r.TGraphAsymmErrors()
ratioPlot.Divide(dataHist, diphoHist, 'pois')
canvB = r.TCanvas()
ratioPlot.GetYaxis().SetRangeUser(0.,2.)
ratioPlot.Draw()
canvB.Print('%s/dataMCdiphoBDT_ratio_%s.png'%(plotDir,plotExt))
canvB.Print('%s/dataMCdiphoBDT_ratio_%s.pdf'%(plotDir,plotExt))

