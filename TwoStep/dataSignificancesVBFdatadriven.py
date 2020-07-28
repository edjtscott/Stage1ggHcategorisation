#usual imports
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
from addRowFunctions import truthVBF, vbfWeight, cosThetaStar, addLeadJetPhi, addSubleadJetPhi, modifyMjjHEM, modifyPtHjjHEM
from catOptim import CatOptim

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Name of dataframe if it already exists')
parser.add_option('-s','--signalFrame', default=None, help='Name of signal dataframe if it already exists')
parser.add_option('-m','--modelName', default=None, help='Name of diphoton model for testing')
parser.add_option('-n','--nIterations', default=2000, help='Number of iterations to run for random significance optimisation')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models')

#define the different sets of variables used
from variableDefinitions import allVarsData, allVarsGen, diphoVars, dijetVars, lumiDict

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'vh':'VH.root'}
theProcs = procFileMap.keys()
dataFileMap = {'Data':'Data.root'}
signals     = ['ggh','vbf','vh']

#define selection
queryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_leadIDMVA>-0.2) and (dipho_subleadIDMVA>-0.2) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>40.) and (dijet_SubJPt>30.) and (dijet_Mjj>350.)'

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

  trainTotal['truthVBF'] = trainTotal.apply(truthVBF,axis=1)
  trainTotal['dijet_centrality'] = np.exp(-4.*((trainTotal.dijet_Zep/trainTotal.dijet_abs_dEta)**2))
  trainTotal['dijet_leadPhi'] = trainTotal.apply(addLeadJetPhi,axis=1)
  trainTotal['dijet_subleadPhi'] = trainTotal.apply(addSubleadJetPhi,axis=1)
  #trainTotal['dijet_Mjj'] = trainTotal.apply(modifyMjjHEM,axis=1) #FIXME testing HEM effect
  #trainTotal['dipho_dijet_ptHjj'] = trainTotal.apply(modifyPtHjjHEM,axis=1)

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
  
  #add needed variables
  dataTotal['dijet_centrality']=np.exp(-4.*((dataTotal.dijet_Zep/dataTotal.dijet_abs_dEta)**2))

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  dataTotal.to_pickle('%s/vbfDataTotal.pkl'%frameDir)
  print 'frame saved as %s/vbfDataTotal.pkl'%frameDir
else:
  dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))

#define the variables used as input to the classifier
vbfDX = trainTotal[diphoVars].values
vbfX  = trainTotal[dijetVars].values
vbfY  = trainTotal['truthVBF'].values
vbfP  = trainTotal['HTXSstage1p2bin'].values
vbfM  = trainTotal['dipho_mass'].values
vbfFW = trainTotal['weight'].values
vbfJ  = trainTotal['dijet_Mjj'].values
vbfH  = trainTotal['dipho_pt'].values
vbfHJ = trainTotal['dipho_dijet_ptHjj'].values
vbfIL = trainTotal['dipho_leadIDMVA'].values
vbfIS = trainTotal['dipho_subleadIDMVA'].values

dataDX = dataTotal[diphoVars].values
dataX  = dataTotal[dijetVars].values
dataM  = dataTotal['dipho_mass'].values
dataFW = np.ones(dataM.shape[0])
dataJ  = dataTotal['dijet_Mjj'].values
dataH  = dataTotal['dipho_pt'].values
dataHJ = dataTotal['dipho_dijet_ptHjj'].values
dataIL = dataTotal['dipho_leadIDMVA'].values
dataIS = dataTotal['dipho_subleadIDMVA'].values

#evaluate dipho BDT scores
diphoMatrix = xg.DMatrix(vbfDX, label=vbfY, weight=vbfFW, feature_names=diphoVars)
diphoDataMatrix  = xg.DMatrix(dataDX,  label=dataFW, weight=dataFW,  feature_names=diphoVars)
diphoModel = xg.Booster()
diphoModel.load_model('%s/%s'%(modelDir,opts.modelName))
vbfD = diphoModel.predict(diphoMatrix)
dataD  = diphoModel.predict(diphoDataMatrix)

#now evaluate the dijet BDT scores
numClasses=3
vbfMatrix = xg.DMatrix(vbfX, label=vbfY, weight=vbfFW, feature_names=dijetVars)
dataMatrix  = xg.DMatrix(dataX,  label=dataFW, weight=dataFW,  feature_names=dijetVars)
vbfModel = xg.Booster()
vbfModel.load_model('%s/%s'%(modelDir,'vbfDataDriven.model'))
vbfPredictions = vbfModel.predict(vbfMatrix).reshape(vbfM.shape[0],numClasses)
vbfV = vbfPredictions[:,2]
vbfG = vbfPredictions[:,1]
print 'some values of the VBF probability %s'%vbfPredictions[0:10,2]
print 'some values of the ggH probability %s'%vbfPredictions[0:10,1]
print 'some values of the bkg probability %s'%vbfPredictions[0:10,0]
dataPredictions = vbfModel.predict(dataMatrix).reshape(dataM.shape[0],numClasses)
dataV  = dataPredictions[:,2]
dataG  = dataPredictions[:,1]
print 'some values of the VBF probability %s'%dataPredictions[0:10,2]
print 'some values of the ggH probability %s'%dataPredictions[0:10,1]
print 'some values of the bkg probability %s'%dataPredictions[0:10,0]

#now estimate significance using the amount of background in a plus/mins 1 sigma window
#set up parameters for the optimiser
ranges = [ [0.3,1.], [0.,0.7], [0.5,1.] ]
#ranges = [ [0.3,1.], [0.,0.2], [0.5,1.] ] #FIXME testing lower ggH bounds
names  = ['VBFscore', 'GGHscore', 'DiphotonBDT']
printStr = ''

## configure the signal and background for VBF

## two categories inclusive
#sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.)
#bkgWeights = dataFW * (dataJ>350.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for VBF with two categories are: \n'
#printStr += optimiser.getPrintableResult()
#
#
### three categories inclusive
#sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.)
#bkgWeights = dataFW * (dataJ>350.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 3, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for VBF with three categories are: \n'
#printStr += optimiser.getPrintableResult()
#
### four categories inclusive
#sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.)
#bkgWeights = dataFW * (dataJ>350.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 4, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for VBF with four categories are: \n'
#printStr += optimiser.getPrintableResult()
#
#
### split by pT(Hjj) only, with BSM bin
#runningTotal = 0.
#sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfH<200.) * (vbfHJ<25.)
#bkgWeights = dataFW * (dataJ>350.) * (dataH<200.) * (dataHJ<25.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH<200.) * (vbfHJ<25.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for the VBF low pTHjj bin are: \n'
#printStr += optimiser.getPrintableResult()
#runningTotal += optimiser.getBests().getTotSignif()**2
#
#sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfH<200.) * (vbfHJ>25.)
#bkgWeights = dataFW * (dataJ>350.) * (dataH<200.) * (dataHJ>25.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH<200.) * (vbfHJ>25.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for the VBF high pTHjj bin are: \n'
#printStr += optimiser.getPrintableResult()
#runningTotal += optimiser.getBests().getTotSignif()**2
#
#sigWeights = vbfFW * (vbfP==206) * (vbfJ>350.) * (vbfH>200.)
#bkgWeights = dataFW * (dataJ>350.) * (dataH>200.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH>200.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 1, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for the VBF BSM bin are: \n'
#printStr += optimiser.getPrintableResult()
#runningTotal += optimiser.getBests().getTotSignif()**2
#
#printStr += 'Which means that the total VBF significance for the pTHjj split plus BSM is : %1.3f \n\n\n'%np.sqrt(runningTotal)
#
#
### split by mjj only, with BSM bin
#runningTotal = 0.
#sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfH<200.) * (vbfJ<700.)
#bkgWeights = dataFW * (dataJ>350.) * (dataH<200.) * (dataJ<700.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH<200.) * (vbfJ<700.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for the VBF low mjj bin are: \n'
#printStr += optimiser.getPrintableResult()
#runningTotal += optimiser.getBests().getTotSignif()**2
#
#sigWeights = vbfFW * (vbfY==2) * (vbfJ>700.) * (vbfH<200.)
#bkgWeights = dataFW * (dataJ>700.) * (dataH<200.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>700.) * (vbfH<200.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for the VBF high mjj bin are: \n'
#printStr += optimiser.getPrintableResult()
#runningTotal += optimiser.getBests().getTotSignif()**2
#
#sigWeights = vbfFW * (vbfP==206) * (vbfJ>350.) * (vbfH>200.)
#bkgWeights = dataFW * (dataJ>350.) * (dataH>200.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH>200.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 1, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for the VBF BSM bin are: \n'
#printStr += optimiser.getPrintableResult()
#runningTotal += optimiser.getBests().getTotSignif()**2
#
#printStr += 'Which means that the total VBF significance for the mjj split plus BSM is : %1.3f \n\n\n'%np.sqrt(runningTotal)


## split by pT(Hjj) and mjj, with BSM bin
runningTotal = 0.
sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfH<200.) * (vbfJ<700.) * (vbfHJ<25.)
bkgWeights = dataFW * (dataJ>350.) * (dataH<200.) * (dataJ<700.) * (dataHJ<25.)
nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH<200.) * (vbfJ<700.) * (vbfHJ<25.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 1, ranges, names)
optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 3, ranges, names)
optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
optimiser.setOpposite('GGHscore')
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for the VBF low pTHjj, low mjj bin are: \n'
printStr += optimiser.getPrintableResult()
runningTotal += optimiser.getBests().getTotSignif()**2

sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfH<200.) * (vbfJ<700.) * (vbfHJ>25.)
bkgWeights = dataFW * (dataJ>350.) * (dataH<200.) * (dataJ<700.) * (dataHJ>25.)
nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH<200.) * (vbfJ<700.) * (vbfHJ>25.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 1, ranges, names)
optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 3, ranges, names)
optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
optimiser.setOpposite('GGHscore')
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for the VBF high pTHjj, low mjj bin are: \n'
printStr += optimiser.getPrintableResult()
runningTotal += optimiser.getBests().getTotSignif()**2

sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfH<200.) * (vbfJ>700.) * (vbfHJ<25.)
bkgWeights = dataFW * (dataJ>350.) * (dataH<200.) * (dataJ>700.) * (dataHJ<25.)
nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH<200.) * (vbfJ>700.) * (vbfHJ<25.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 1, ranges, names)
optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 3, ranges, names)
optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
optimiser.setOpposite('GGHscore')
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for the VBF low pTHjj, high mjj bin are: \n'
printStr += optimiser.getPrintableResult()
runningTotal += optimiser.getBests().getTotSignif()**2

sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfH<200.) * (vbfJ>700.) * (vbfHJ>25.)
bkgWeights = dataFW * (dataJ>350.) * (dataH<200.) * (dataJ>700.) * (dataHJ>25.)
nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH<200.) * (vbfJ>700.) * (vbfHJ>25.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 1, ranges, names)
optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 3, ranges, names)
optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
optimiser.setOpposite('GGHscore')
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for the VBF high pTHjj, high mjj bin are: \n'
printStr += optimiser.getPrintableResult()
runningTotal += optimiser.getBests().getTotSignif()**2

sigWeights = vbfFW * (vbfP==206) * (vbfJ>350.) * (vbfH>200.)
bkgWeights = dataFW * (dataJ>350.) * (dataH>200.)
nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfH>200.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 1, ranges, names)
optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 2, ranges, names)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 3, ranges, names)
optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
optimiser.setOpposite('GGHscore')
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for the VBF BSM bin are: \n'
printStr += optimiser.getPrintableResult()
runningTotal += optimiser.getBests().getTotSignif()**2

printStr += 'Which means that the total VBF significance for the pTHjj and mjj splits, plus BSM is : %1.3f \n\n\n'%np.sqrt(runningTotal)


## configure the signal and background for VBF-like ggH
## test one inclusive cat here

#print 'ED DEBUG understanding ggH score'
#for score in np.arange(0.04, 0.99, 0.05):
#  sigVal = 0.68 * np.sum( vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfV<0.4) * (vbfG > score) )
#  nonVal = 0.68 * np.sum( vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfV<0.4) * (vbfG > score) )
#  bkgVal = (4./80.) * np.sum( dataFW * (dataJ>350.) * (dataV<0.4) * (dataG > score) )
#  if bkgVal+sigVal+nonVal > 0.:
#    tempSignif = sigVal/np.sqrt(bkgVal+sigVal+nonVal)
#  else: tempSignif = 0.
#  print 'for score of %.3f the ggH, VBF, bkg, S/sqrt(S+B) values are %.3f, %.3f, %.3f, %.3f'%(score, sigVal, nonVal, bkgVal, tempSignif)

names  = ['GGHscore', 'VBFscore', 'DiphotonBDT']
#sigWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfV<0.4)
#bkgWeights = dataFW * (dataJ>350.) * (dataV<0.4)
#nonWeights = vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfV<0.4) 
#optimiser = CatOptim(sigWeights, vbfM, [vbfG,vbfV,vbfD], bkgWeights, dataM, [dataG,dataV,dataD], 1, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfG,vbfV,vbfD])
#optimiser.setOpposite('VBFscore')
#optimiser.setConstBkg(True)
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for ggH VBF-like with on inclusive category are: \n'
#printStr += optimiser.getPrintableResult()
#
### test two inclusive cats here
sigWeights = vbfFW * (vbfY==1) * (vbfJ>350.) * (vbfV<0.4)
bkgWeights = dataFW * (dataJ>350.) * (dataV<0.4)
nonWeights = vbfFW * (vbfY==2) * (vbfJ>350.) * (vbfV<0.4) 
optimiser = CatOptim(sigWeights, vbfM, [vbfG,vbfV,vbfD], bkgWeights, dataM, [dataG,dataV,dataD], 2, ranges, names)
optimiser.setNonSig(nonWeights, vbfM, [vbfG,vbfV,vbfD])
optimiser.setOpposite('VBFscore')
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for ggH VBF-like with two inclusive categories are: \n'
printStr += optimiser.getPrintableResult()

print
print printStr
print
