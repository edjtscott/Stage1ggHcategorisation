#usual imports
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
from addRowFunctions import truthVhHad, vhHadWeight, cosThetaStar
from catOptim import CatOptim

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Name of dataframe if it already exists')
parser.add_option('-s','--signalFrame', default=None, help='Name of signal dataframe if it already exists')
parser.add_option('-m','--modelName', default=None, help='Name of diphoton model for testing')
parser.add_option('-n','--nIterations', default=5000, help='Number of iterations to run for random significance optimisation')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models')

#define the different sets of variables used
from variableDefinitions import allVarsData, allVarsGen, diphoVars, vhHadVars, lumiDict

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'vh':'VH.root'}
theProcs = procFileMap.keys()
dataFileMap = {'Data':'Data.root'}
signals     = ['ggh','vbf','vh']

#define selection
queryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>30.) and (dijet_SubJPt>30.) and (dijet_Mjj>60.) and (dijet_Mjj<120.) and (dijet_leadEta>-2.4) and (dijet_subleadEta>-2.4) and (dijet_leadEta<2.4) and (dijet_subleadEta<2.4) and (dipho_leadIDMVA>-0.2) and (dipho_subleadIDMVA>-0.2)'

#setup signal frame
trainTotal = None
if opts.signalFrame: 
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.signalFrame))
  print 'Successfully loaded the dataframe'
else:
  trainList = []
  #get trees from files, put them in data frames
  if not 'all' in trainDir:
    for proc,fn in procFileMap.iteritems():
      print 'reading in tree from file %s'%fn
      trainFile   = upr.open('%s/%s'%(trainDir,fn))
      if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
      elif proc in backgrounds: trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
      else: raise Exception('Error did not recognise process %s !'%proc)
      tempFrame = trainTree.pandas.df(allVarsGen).query(queryString)
      tempFrame['proc'] = proc
      trainList.append(tempFrame)
  else:
    for year in lumiDict.keys():
      for proc,fn in procFileMap.iteritems():
        thisDir = trainDir.replace('all',year)
        print 'reading in tree from file %s'%fn
        trainFile   = upr.open('%s/%s'%(thisDir,fn))
        if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
        elif proc in backgrounds: trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
        else: raise Exception('Error did not recognise process %s !'%proc)
        tempFrame = trainTree.pandas.df(allVarsGen).query(queryString)
        tempFrame['proc'] = proc
        tempFrame.loc[:, 'weight'] = tempFrame['weight'] * lumiDict[year]
        trainList.append(tempFrame)
  print 'got trees and applied selections'

  #create one total frame
  trainTotal = pd.concat(trainList, sort=False)
  del trainList
  del tempFrame
  print 'created total frame'

  trainTotal['truthVhHad'] = trainTotal.apply(truthVhHad,axis=1)

dataTotal = None
if not opts.dataFrame:
  dataList = []
  #get the trees, turn them into arrays
  if not 'all' in trainDir:
    for proc,fn in dataFileMap.iteritems():
      dataFile = upr.open('%s/%s'%(trainDir,fn))
      dataTree = dataFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
      tempData = dataTree.pandas.df(allVarsData).query(queryString)
      tempData['proc'] = proc
      dataList.append(tempData)
  else:
    for year in lumiDict.keys():
      for proc,fn in dataFileMap.iteritems():
        thisDir = trainDir.replace('all',year)
        dataFile = upr.open('%s/%s'%(thisDir,fn))
        dataTree = dataFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
        tempData = dataTree.pandas.df(allVarsData).query(queryString)
        tempData['proc'] = proc
        dataList.append(tempData)
  print 'got trees'

  dataTotal = pd.concat(dataList)

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  dataTotal.to_pickle('%s/vhHadDataTotal.pkl'%frameDir)
  print 'frame saved as %s/vhHadDataTotal.pkl'%frameDir
else:
  dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))

#define the variables used as input to the classifier
vhHadDX = trainTotal[diphoVars].values
vhHadX  = trainTotal[vhHadVars].values
vhHadY  = trainTotal['truthVhHad'].values
vhHadM  = trainTotal['dipho_mass'].values
vhHadFW = trainTotal['weight'].values
vhHadC  = trainTotal['cosThetaStar'].values
vhHadP  = trainTotal['HTXSstage1p2bin'].values

dataDX = dataTotal[diphoVars].values
dataX  = dataTotal[vhHadVars].values
dataM  = dataTotal['dipho_mass'].values
dataFW = np.ones(dataM.shape[0])
dataC  = dataTotal['cosThetaStar'].values

#evaluate dipho BDT scores
diphoMatrix = xg.DMatrix(vhHadDX, label=vhHadY, weight=vhHadFW, feature_names=diphoVars)
diphoDataMatrix  = xg.DMatrix(dataDX,  label=dataFW, weight=dataFW,  feature_names=diphoVars)
diphoModel = xg.Booster()
diphoModel.load_model('%s/%s'%(modelDir,opts.modelName))
vhHadD = diphoModel.predict(diphoMatrix)
dataD  = diphoModel.predict(diphoDataMatrix)

#now evaluate the VH had BDT scores
vhHadMatrix = xg.DMatrix(vhHadX, label=vhHadY, weight=vhHadFW, feature_names=vhHadVars)
dataMatrix  = xg.DMatrix(dataX,  label=dataFW, weight=dataFW,  feature_names=vhHadVars)
vhHadModel = xg.Booster()
vhHadModel.load_model('%s/%s'%(modelDir,'vhHadDataDriven.model'))
vhHadV = vhHadModel.predict(vhHadMatrix)
dataV  = vhHadModel.predict(dataMatrix)

#now estimate significance using the amount of background in a plus/mins 1 sigma window
#set up parameters for the optimiser
ranges = [ [0.,1.], [0.5,1.] ]
names  = ['VHhadBDT','DiphotonBDT']
printStr = ''

#configure the signal and background
sigWeights = vhHadFW * (vhHadY>0.5)
bkgWeights = dataFW
nonWeights = vhHadFW * (vhHadP>99.5) * (vhHadP<116.5)
optimiser = CatOptim(sigWeights, vhHadM, [vhHadV,vhHadD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.setNonSig(nonWeights, vhHadM, [vhHadV,vhHadD])
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for VH hadronic bin are: \n'
printStr += optimiser.getPrintableResult()

ranges = [ [0.5,1.] ]
names  = ['DiphotonBDT']
sigWeights = vhHadFW * (vhHadY>0.5) * (vhHadC<0.5) * (vhHadC>-0.5)
bkgWeights = dataFW * (dataC<0.5) * (dataC>-0.5)
nonWeights = vhHadFW * (vhHadP>99.5) * (vhHadP<116.5) * (vhHadC<0.5) * (vhHadC>-0.5)
optimiser = CatOptim(sigWeights, vhHadM, [vhHadD], bkgWeights, dataM, [dataD], 2, ranges, names)
optimiser.setNonSig(nonWeights, vhHadM, [vhHadD])
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results using ONLY diphoton BDT are: \n'
printStr += optimiser.getPrintableResult()

print
print printStr
print
