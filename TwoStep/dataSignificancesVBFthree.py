#usual imports
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
from addRowFunctions import truthVBF, vbfWeight
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
              'cosThetaStar', 'dipho_cosphi', 'vtxprob', 'sigmarv', 'sigmawv', 'weight', 'dipho_mass', 'dijet_dphi', 'dijet_minDRJetPho', 'dijet_Zep']

diphoVars = ['dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM',
              'dijet_leadEta', 'dijet_subleadEta', 
              'dipho_cosphi', 'vtxprob', 'sigmarv', 'sigmawv']

dijetVars = ['dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_LeadJPt', 'dijet_SubJPt', 'dijet_abs_dEta', 'dijet_Mjj', 'dijet_centrality', 'dijet_dphi', 'dijet_minDRJetPho', 'dijet_dipho_dphi_trunc']

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
  dataTotal = dataTotal[dataTotal.dijet_LeadJPt>40.]
  dataTotal = dataTotal[dataTotal.dijet_SubJPt>30.]
  dataTotal = dataTotal[dataTotal.dijet_Mjj>250.]
  print 'done jet cuts'

  #add needed variables
  dataTotal['dijet_centrality']=np.exp(-4.*((dataTotal.dijet_Zep/dataTotal.dijet_abs_dEta)**2))

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  dataTotal.to_pickle('%s/dataTotal.pkl'%frameDir)
  print 'frame saved as %s/dataTotal.pkl'%frameDir
else:
  dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))

#define the variables used as input to the classifier
vbfDX = trainTotal[diphoVars].values
vbfX  = trainTotal[dijetVars].values
vbfY  = trainTotal['truthVBF'].values
vbfP  = trainTotal['HTXSstage1_1_cat'].values
vbfM  = trainTotal['dipho_mass'].values
vbfFW = trainTotal['weight'].values
vbfJ  = trainTotal['dijet_Mjj'].values

dataDX = dataTotal[diphoVars].values
dataX  = dataTotal[dijetVars].values
dataM  = dataTotal['dipho_mass'].values
dataFW = np.ones(dataM.shape[0])
dataJ  = dataTotal['dijet_Mjj'].values

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
vbfModel.load_model('%s/%s'%(modelDir,'vbfModel.model'))
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
ranges = [ [0.,1.], [0.,1.], [0.,1.] ]
names  = ['VBFscore', 'GGHscore', 'DiphotonBDT']
printStr = ''

#configure the signal and background
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
#sigWeights = vbfFW * (vbfY==2) * (vbfJ>350.)
#bkgWeights = dataFW * (dataJ>350.)
#nonWeights = vbfFW * (vbfY==1) * (vbfJ>350.)
#optimiser = CatOptim(sigWeights, vbfM, [vbfV,vbfG,vbfD], bkgWeights, dataM, [dataV,dataG,dataD], 3, ranges, names)
#optimiser.setNonSig(nonWeights, vbfM, [vbfV,vbfG,vbfD])
#optimiser.setOpposite('GGHscore')
#optimiser.optimise(opts.intLumi, opts.nIterations)
#printStr += 'Results for VBF with three categories are: \n'
#printStr += optimiser.getPrintableResult()

#configure the signal and background
names  = ['GGHscore', 'VBFscore', 'DiphotonBDT']
sigWeights = vbfFW * (vbfP>109.5) * (vbfP<113.5) * (vbfJ>350.) * (vbfV<0.4)
bkgWeights = dataFW * (dataJ>350. * (dataV<0.4))
nonWeights = vbfFW * (vbfP>206.5) * (vbfP<211.5) * (vbfJ>350.) * (vbfV<0.4) 
optimiser = CatOptim(sigWeights, vbfM, [vbfG,vbfV,vbfD], bkgWeights, dataM, [dataG,dataV,dataD], 1, ranges, names)
optimiser.setNonSig(nonWeights, vbfM, [vbfG,vbfV,vbfD])
optimiser.setOpposite('VBFscore')
optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for ggH 2J-like are: \n'
printStr += optimiser.getPrintableResult()

print
print printStr
print
