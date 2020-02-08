#usual imports
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
from addRowFunctions import truthVhHad, vhHadWeight

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

#including the full selection
hdfQueryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>30.) and (dijet_SubJPt>30.) and (dijet_Mjj>60.) and (dijet_Mjj<120.) and (dijet_leadEta>-2.4) and (dijet_subleadEta>-2.4) and (dijet_leadEta<2.4) and (dijet_subleadEta<2.4)'
queryString = hdfQueryString + ' and (dipho_leadIDMVA>-0.2) and (dipho_subleadIDMVA>-0.2)'

#define hdf input
hdfDir = trainDir.replace('trees','hdfs')

if hdfDir.count('all'):
  #hdfFrame = pd.read_hdf('%s/VH_with_DataDriven_2016.h5'%hdfDir).query(hdfQueryString)
  #hdfFrame.append( pd.read_hdf('%s/VH_with_DataDriven_2017.h5'%hdfDir).query(hdfQueryString) )
  #hdfFrame.append( pd.read_hdf('%s/VH_with_DataDriven_2018.h5'%hdfDir).query(hdfQueryString) )
  #hdfFrame = pd.read_hdf('%s/VH_with_DataDriven_2018.h5'%hdfDir).query(hdfQueryString)
  hdfFrame = pd.read_hdf('%s/VH_with_DataDriven_2018_MERGEDFF.h5'%hdfDir).query(hdfQueryString)
else:
  hdfFrame = pd.read_hdf('%s/VH_with_DataDriven_%s.h5'%(hdfDir,hdfDir.split('/')[-2]) ).query(hdfQueryString)

hdfFrame = hdfFrame[hdfFrame['sample']=='QCD']
hdfFrame['proc'] = 'datadriven'
print 'ED DEBUG sum of datadriven weights %.3f'%np.sum(hdfFrame['weight'].values)

#define input files
procFileMap = {'ggh':'powheg_ggH.root', 'vbf':'powheg_VBF.root', 'vh':'VH.root',
               'dipho':'Dipho.root'}
theProcs = procFileMap.keys()
signals     = ['ggh','vbf','vh']
backgrounds = ['dipho']

#define variables to be used
from variableDefinitions import allVarsGen, vhHadVars

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get trees from files, put them in data frames
  for proc,fn in procFileMap.iteritems():
      print 'reading in tree from file %s'%fn
      trainFile   = upr.open('%s/%s'%(trainDir,fn))
      if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
      elif proc in backgrounds: trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
      else: raise Exception('Error did not recognise process %s !'%proc)
      trainFrames[proc] = trainTree.pandas.df(allVarsGen).query(queryString)
      trainFrames[proc]['proc'] = proc
  print 'got trees and applied selections'

  #create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  trainList.append(hdfFrame)
  trainTotal = pd.concat(trainList, sort=False)
  del trainFrames
  del hdfFrame
  print 'created total frame'

  #add the target variable and the equalised weight
  trainTotal['truthVhHad'] = trainTotal.apply(truthVhHad,axis=1)
  sigSumW = np.sum(trainTotal[trainTotal.truthVhHad==1]['weight'].values)
  bkgSumW = np.sum(trainTotal[trainTotal.truthVhHad==0]['weight'].values)
  print 'sigSumW, bkgSumW, ratio = %.3f, %.3f, %.3f'%(sigSumW, bkgSumW, sigSumW/bkgSumW)
  trainTotal['vhHadWeight'] = trainTotal.apply(vhHadWeight, axis=1, args=[bkgSumW/sigSumW])
  #trainTotal = trainTotal[trainTotal.truthVhHad>-0.5] ## this is left over from when only training VH vs ggH

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/vhHadDataDriven.pkl'%frameDir)
  print 'frame saved as %s/vhHadDataDriven.pkl'%frameDir

#read in dataframe if above steps done before
else:
  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
  print 'Successfully loaded the dataframe'

#set up train set and randomise the inputs
trainFrac = 0.8
theShape = trainTotal.shape[0]
theShuffle = np.random.permutation(theShape)
trainLimit = int(theShape*trainFrac)

#define the values needed for training as numpy arrays
vhHadX  = trainTotal[vhHadVars].values
vhHadY  = trainTotal['truthVhHad'].values
vhHadTW = trainTotal['vhHadWeight'].values
vhHadFW = trainTotal['weight'].values
vhHadM  = trainTotal['dipho_mass'].values

#do the shuffle
vhHadX  = vhHadX[theShuffle]
vhHadY  = vhHadY[theShuffle]
vhHadTW = vhHadTW[theShuffle]
vhHadFW = vhHadFW[theShuffle]
vhHadM  = vhHadM[theShuffle]

#split into train and test
vhHadTrainX,  vhHadTestX  = np.split( vhHadX,  [trainLimit] )
vhHadTrainY,  vhHadTestY  = np.split( vhHadY,  [trainLimit] )
vhHadTrainTW, vhHadTestTW = np.split( vhHadTW, [trainLimit] )
vhHadTrainFW, vhHadTestFW = np.split( vhHadFW, [trainLimit] )
vhHadTrainM,  vhHadTestM  = np.split( vhHadM,  [trainLimit] )

#set up the training and testing matrices
trainMatrix = xg.DMatrix(vhHadTrainX, label=vhHadTrainY, weight=vhHadTrainTW, feature_names=vhHadVars)
testMatrix  = xg.DMatrix(vhHadTestX, label=vhHadTestY, weight=vhHadTestFW, feature_names=vhHadVars)
trainParams = {}
trainParams['objective'] = 'binary:logistic'
trainParams['nthread'] = 1
#trainParams['seed'] = 123456

#add any specified training parameters
paramExt = ''
if opts.trainParams:
  paramExt = '__'
  for pair in opts.trainParams:
    key  = pair.split(':')[0]
    data = pair.split(':')[1]
    trainParams[key] = data
    paramExt += '%s_%s__'%(key,data)
  paramExt = paramExt[:-2]

#train the BDT
print 'about to train diphoton BDT'
vhHadModel = xg.train(trainParams, trainMatrix)
print 'done'

#save it
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
vhHadModel.save_model('%s/vhHadDataDriven%s.model'%(modelDir,paramExt))
print 'saved as %s/vhHadDataDrivenl%s.model'%(modelDir,paramExt)

#evaluate performance using area under the ROC curve
vhHadPredYtrain = vhHadModel.predict(trainMatrix)
vhHadPredYtest  = vhHadModel.predict(testMatrix)
print 'Training performance:'
print 'area under roc curve for training set = %1.3f'%( roc_auc_score(vhHadTrainY, vhHadPredYtrain, sample_weight=vhHadTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( roc_auc_score(vhHadTestY,  vhHadPredYtest,  sample_weight=vhHadTestFW)  )

#check yields for various working points
testScale = 1./(1.-trainFrac)
for cutVal in np.arange(0.1,0.96,0.05):
  selectedSig = opts.intLumi * testScale * np.sum( vhHadTestFW * (vhHadTestY==1) * (vhHadPredYtest>cutVal) )
  selectedBkg = opts.intLumi * testScale * np.sum( vhHadTestFW * (vhHadTestY==0) * (vhHadPredYtest>cutVal) )
  print 'Selected events for a cut value of %.2f: S %.3f, B %.3f'%(cutVal, selectedSig, selectedBkg)
  selectedSigWindow = opts.intLumi * testScale * np.sum( vhHadTestFW * (vhHadTestY==1) * (vhHadPredYtest>cutVal) * (vhHadTestM>123.) * (vhHadTestM<127.) )
  selectedBkgWindow = opts.intLumi * testScale * np.sum( vhHadTestFW * (vhHadTestY==0) * (vhHadPredYtest>cutVal) * (vhHadTestM>123.) * (vhHadTestM<127.) )
  print 'and applying a 2 GeV mass window w/%.2f: S %.3f, B %.3f'%(cutVal, selectedSigWindow, selectedBkgWindow)
