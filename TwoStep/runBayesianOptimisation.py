#usual imports
import ROOT as r
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

from addRowFunctions import truthVhHad, vhHadWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import fill_hist
import usefulStyle as useSty

from matplotlib import rc
from bayes_opt import BayesianOptimization




pd.options.mode.chained_assignment = None

np.random.seed(42)



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
if trainDir.endswith('/'): trainDir = trainDir[:-1]# slice the string to remove the last character i.e the "/"
frameDir = trainDir.replace('trees','frames')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')#separate train options based on comma (used to define parameter pairs)

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'vh':'VH.root'}# a dictionary with file names
theProcs = procFileMap.keys()# list of keys i.e 'ggh','vbf','vh'


allVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar', 'dipho_cosphi', 'vtxprob','sigmarv','sigmawv','weight', 'tempStage1bin','dipho_mass','dipho_leadEta','dipho_subleadEta','cos_dijet_dipho_dphi','dijet_Zep','dijet_jet1_QGL','dijet_jet2_QGL']


#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:#if the dataframe option was not used while running, create dataframe from files in folder
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():#proc, fn are the pairs 'proc':'fn' in the file map 
      trainFile   = upr.open('%s/%s'%(trainDir,fn))

  #is a reader and a writer of the ROOT file format using only Python and Numpy.
  #Unlike PyROOT and root_numpy, uproot does not depend on C++ ROOT. Instead, it uses Numpy to cast blocks of data from the ROOT file as Numpy arrays.
      trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc]
      trainFrames[proc] = trainTree.pandas.df(allVars)
      trainFrames[proc]['proc'] = proc #adding a column for the process
  print 'got trees'

#create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'


#then filter out the events into only those with the phase space we are interested in
  trainTotal = trainTotal[trainTotal.dipho_mass>100.]# diphoton mass range
  trainTotal = trainTotal[trainTotal.dipho_mass<180.]# diphoton mass range
  print 'done mass cuts'
#some extra cuts that are applied for vhHad BDT in the AN
  trainTotal = trainTotal[trainTotal.dipho_leadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_subleadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_lead_ptoM>0.333]
  trainTotal = trainTotal[trainTotal.dipho_sublead_ptoM>0.25]
  print 'done basic preselection cuts'
#cut on the jet pT to require at least 2 jets
  trainTotal = trainTotal[trainTotal.dijet_LeadJPt>30.]
  trainTotal = trainTotal[trainTotal.dijet_SubJPt>30.]
  print 'done jet pT cuts'
#consider the VH hadronic mjj region (ideally to cut on gen mjj for this)
  trainTotal = trainTotal[trainTotal.dijet_Mjj>60.]
  trainTotal = trainTotal[trainTotal.dijet_Mjj<120.]
  print 'done mjj cuts'


#adding variables that need to be calculated

  trainTotal['dijet_dipho_dEta']=((trainTotal.dijet_leadEta+trainTotal.dijet_subleadEta)/2)-((trainTotal.dipho_leadEta+trainTotal.dipho_subleadEta)/2)
  trainTotal['dijet_centrality_gg']=np.exp(-4*(trainTotal.dijet_Zep/trainTotal.dijet_abs_dEta)**2)
  print 'calculated variables added'


#add the target variable and the equalised weight
  trainTotal['truthVhHad'] = trainTotal.apply(truthVhHad,axis=1)#the truthVhHad function returns 1 if the gen HTXS stage 0 category of the event id VH hadronic
  sigSumW = np.sum(trainTotal[trainTotal.truthVhHad==1]['weight'].values)#summing weights of vh hadronic events (selected by gen HTXS category)
  bkgSumW = np.sum(trainTotal[trainTotal.truthVhHad==0]['weight'].values)#summing weights of non-vh hadronic events (selected by gen HTXS stage 0 category
  print 'sigSumW, bkgSumW, ratio = %.3f, %.3f, %.3f'%(sigSumW, bkgSumW, sigSumW/bkgSumW)

  trainTotal['vhHadWeight'] = trainTotal.apply(vhHadWeight, axis=1, args=[bkgSumW/sigSumW])#multiply each of the VH weight values by sum of VH weight/sum of non vh weight -- to check

trainTotal = trainTotal[trainTotal.truthVhHad>-0.5]



#save as a pickle file
#if not path.isdir(frameDir): 
system('mkdir -p %s'%frameDir)
trainTotal.to_pickle('%s/vhHadTotal.pkl'%frameDir)
print 'frame saved as %s/vhHadTotal.pkl'%frameDir

#read in dataframe if above steps done before
#else:
 # trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
print 'Successfully loaded the dataframe'

#set up train set and randomise the inputs
trainFrac = 0.8

theShape = trainTotal.shape[0]#number of rows in total dataframe
theShuffle = np.random.permutation(theShape)
trainLimit = int(theShape*trainFrac)


#define the values needed for training as numpy arrays
#vhHadVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar','cos_dijet_dipho_dphi', 'dijet_dipho_dEta']#do not provide dipho_mass=>do not bias the BDT by the Higgs mass used in signal MC

vhHadVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva','dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_nj', 'cosThetaStar', 'cos_dijet_dipho_dphi','dijet_dipho_dEta','dijet_centrality_gg']#,'dijet_jet1_QGL','dijet_jet2_QGL']



vhHadX  = trainTotal[vhHadVars].values# the train input variables defined in the above list
vhHadY  = trainTotal['truthVhHad'].values#the training target two classes 1 for vh had 0 for other processes 
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


early_stops = []
def train_clf(min_child_weight, colsample_bytree, max_depth, subsample, gamma, reg_alpha, reg_lambda):
        res = xg.cv(
            {
                'min_child_weight': min_child_weight,
                'colsample_bytree': colsample_bytree, 
                'max_depth': int(max_depth),
                'subsample': subsample,
                'gamma': gamma,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'objective' : 'binary:logistic', 
                'tree_method' : 'hist',
                'eval_metric': 'auc'            
            },
            trainMatrix, num_boost_round=2000, nfold=5, early_stopping_rounds=100
        )
        early_stops.append(len(res))

        value = res['test-auc-mean'].iloc[-1]
    
        return value

hyperpar =  {
        'min_child_weight': (0, 10),
        'colsample_bytree': (0.5, 1),
        'max_depth': (4, 20),
        'subsample': (0.8, 1),
        'gamma': (0, 7),
        'reg_alpha': (15, 30),
        'reg_lambda': (15, 30)
    }

xgb_bo = BayesianOptimization(train_clf, hyperpar)    
xgb_bo.maximize(init_points=3, n_iter=20, acq='ei')
params = xgb_bo.max['params']

print 'Bayesian Parameters'

print params
