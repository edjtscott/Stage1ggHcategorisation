#usual imports
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
from addRowFunctions import truthVhHad, vhHadWeight
from catOptim import CatOptim

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Name of dataframe if it already exists')
parser.add_option('-s','--signalFrame', default=None, help='Name of signal dataframe if it already exists')
parser.add_option('-m','--modelName', default=None, help='Name of diphoton model for testing')
parser.add_option('-n','--nIterations', default=10000, help='Number of iterations to run for random significance optimisation')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models')

#define the different sets of variables used
allVars    = ['dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM',
              'dijet_leadEta', 'dijet_subleadEta', 'dijet_LeadJPt', 'dijet_SubJPt', 'dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'dipho_dijet_ptHjj', 'dijet_dipho_dphi_trunc',
              'cosThetaStar', 'dipho_cosphi', 'vtxprob', 'sigmarv', 'sigmawv', 'weight', 'dipho_mass']

diphoVars = ['dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM',
              'dijet_leadEta', 'dijet_subleadEta', 
              'dipho_cosphi', 'vtxprob', 'sigmarv', 'sigmawv']

vhHadVars  = ['dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_leadEta', 'dijet_subleadEta', 'dijet_LeadJPt', 'dijet_SubJPt', 'dijet_Mjj', 'dijet_abs_dEta', 'cosThetaStar']

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'vh':'VH.root'}
theProcs = procFileMap.keys()
dataFileMap = {'Data':'Data.root'}

trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.signalFrame))
print 'Successfully loaded the dataframe'

dataTotal = None
if not opts.dataFrame:
  dataFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in dataFileMap.iteritems():
    dataFile = upr.open('%s/%s'%(trainDir,fn))
    dataTree = dataFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
    dataFrames[proc] = dataTree.pandas.df(allVars)
    dataFrames[proc]['proc'] = proc
  print 'got trees'

  dataTotal = dataFrames['Data']
  
  #then filter out the events into only those with the phase space we are interested in
  dataTotal = dataTotal[dataTotal.dipho_mass>100.]
  dataTotal = dataTotal[dataTotal.dipho_mass<180.]
  print 'done mass cuts'
  dataTotal = dataTotal[dataTotal.dipho_leadIDMVA>-0.2]
  dataTotal = dataTotal[dataTotal.dipho_subleadIDMVA>-0.2]
  dataTotal = dataTotal[dataTotal.dipho_lead_ptoM>0.333]
  dataTotal = dataTotal[dataTotal.dipho_sublead_ptoM>0.25]
  print 'done basic preselection cuts'
  dataTotal = dataTotal[dataTotal.dijet_LeadJPt>30.]
  dataTotal = dataTotal[dataTotal.dijet_SubJPt>30.]
  dataTotal = dataTotal[dataTotal.dijet_Mjj>60.]
  dataTotal = dataTotal[dataTotal.dijet_Mjj<120.]
  print 'done jet cuts'

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  dataTotal.to_pickle('%s/dataTotal.pkl'%frameDir)
  print 'frame saved as %s/dataTotal.pkl'%frameDir
else:
  dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))

#define the variables used as input to the classifier
vhHadDX = trainTotal[diphoVars].values
vhHadX  = trainTotal[vhHadVars].values
vhHadY  = trainTotal['truthVhHad'].values
vhHadM  = trainTotal['dipho_mass'].values
vhHadFW = trainTotal['weight'].values

dataDX = dataTotal[diphoVars].values
dataX  = dataTotal[vhHadVars].values
dataM  = dataTotal['dipho_mass'].values
dataFW = np.ones(dataM.shape[0])

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
vhHadModel.load_model('%s/%s'%(modelDir,'vhHadModel.model'))
vhHadV = vhHadModel.predict(vhHadMatrix)
dataV  = vhHadModel.predict(dataMatrix)

#now estimate significance using the amount of background in a plus/mins 1 sigma window
#set up parameters for the optimiser
ranges = [ [0.,1.], [0.,1.] ]
names  = ['VHhadBDT','DiphotonBDT']
printStr = ''

#configure the signal and background
sigWeights = vhHadFW * (vhHadY>0.5)
bkgWeights = dataFW
nonWeights = vhHadFW * (vhHadY<0.5)
optimiser = CatOptim(sigWeights, vhHadM, [vhHadV,vhHadD], bkgWeights, dataM, [dataV,dataD], 2, ranges, names)
optimiser.setNonSig(nonWeights, vhHadM, [vhHadV,vhHadD])
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for VH hadronic bin are: \n'
printStr += optimiser.getPrintableResult()

print
print printStr
print
