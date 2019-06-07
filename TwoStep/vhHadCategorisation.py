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

#define input files
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'vh':'VH.root'}
theProcs = procFileMap.keys()

#define variables to be used
allVars    = ['dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dipho_mva', 
              'dijet_leadEta', 'dijet_subleadEta', 'dijet_LeadJPt', 'dijet_SubJPt', 'dijet_abs_dEta', 'dijet_Mjj', 'dipho_dijet_ptHjj', 'dijet_dipho_dphi_trunc',
              'cosThetaStar', 'dipho_cosphi', 'vtxprob', 'sigmarv', 'sigmawv', 'weight', 'HTXSstage0cat', 'HTXSstage1_1_cat', 'dipho_mass']

vhHadVars  = ['dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_leadEta', 'dijet_subleadEta', 'dijet_LeadJPt', 'dijet_SubJPt', 'dijet_Mjj', 'dijet_abs_dEta', 'cosThetaStar']

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get trees from files, put them in data frames
  for proc,fn in procFileMap.iteritems():
      trainFile   = upr.open('%s/%s'%(trainDir,fn))
      trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc]
      trainFrames[proc] = trainTree.pandas.df(allVars)
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
  trainTotal = trainTotal[trainTotal.dipho_mass>100.]
  trainTotal = trainTotal[trainTotal.dipho_mass<180.]
  print 'done mass cuts'
  trainTotal = trainTotal[trainTotal.dipho_leadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_subleadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_lead_ptoM>0.333]
  trainTotal = trainTotal[trainTotal.dipho_sublead_ptoM>0.25]
  print 'done basic preselection cuts'
  trainTotal = trainTotal[trainTotal.dijet_LeadJPt>30.]
  trainTotal = trainTotal[trainTotal.dijet_SubJPt>30.]
  trainTotal = trainTotal[trainTotal.dijet_Mjj>60.]
  trainTotal = trainTotal[trainTotal.dijet_Mjj<120.]
  print 'done jet cuts'

  #add the target variable and the equalised weight
  trainTotal['truthVhHad'] = trainTotal.apply(truthVhHad,axis=1)
  sigSumW = np.sum(trainTotal[trainTotal.truthVhHad>0.5]['weight'].values)
  bkgSumW = np.sum(trainTotal[trainTotal.truthVhHad<0.5]['weight'].values)
  print 'sigSumW, bkgSumW, ratio = %.3f, %.3f, %.3f'%(sigSumW, bkgSumW, sigSumW/bkgSumW)
  trainTotal['vhHadWeight'] = trainTotal.apply(vhHadWeight, axis=1, args=[bkgSumW/sigSumW])

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/vhHadTotal.pkl'%frameDir)
  print 'frame saved as %s/vhHadTotal.pkl'%frameDir

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

#train the BDT
print 'about to train diphoton BDT'
vhHadModel = xg.train(trainParams, trainMatrix)
print 'done'

#evaluate performance using area under the ROC curve
vhHadPredYtrain = vhHadModel.predict(trainMatrix)
vhHadPredYtest  = vhHadModel.predict(testMatrix)
print 'Training performance:'
print 'area under roc curve for training set = %1.3f'%( roc_auc_score(vhHadTrainY, vhHadPredYtrain, sample_weight=vhHadTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( roc_auc_score(vhHadTestY,  vhHadPredYtest,  sample_weight=vhHadTestFW)  )

#check yields for various working points
testScale = 1./(1.-trainFrac)
for cutVal in np.arange(0.1,0.91,0.05):
  selectedSig = opts.intLumi * testScale * np.sum( vhHadTestFW * (vhHadTestY==1) * (vhHadPredYtest>cutVal) )
  selectedBkg = opts.intLumi * testScale * np.sum( vhHadTestFW * (vhHadTestY==0) * (vhHadPredYtest>cutVal) )
  print 'Selected events for a cut value of %.2f: S %.3f, B %.3f'%(cutVal, selectedSig, selectedBkg)
