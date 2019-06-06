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

from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import fill_hist
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
if trainDir.endswith('/'): trainDir = trainDir[:-1]# slice the string to remove the last character i.e the "/"
frameDir = trainDir.replace('trees','frames')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')#separate train options based on comma (used to define parameter pairs)
trainFrac = 0.7
validFrac = 0.1

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'vh':'VH.root'}# a dictionary with file names
theProcs = procFileMap.keys()# list of keys i.e 'ggh','vbf','vh'

#define the different sets of variables used; will want to revise this for VH
#allVars    = ['leadmva','subleadmva','leadptom','subleadptom', 'leadeta','subleadeta','CosPhi','vtxprob','sigmarv','sigmawv','weight', 'CMS_hgg_mass', 'HTXSstage0cat', 'HTXSstage1_1_cat', 'cosThetaStar']

allVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar', 'dipho_cosphi', 'vtxprob','sigmarv','sigmawv','weight', 'HTXSstage0cat', 'HTXSstage1_1_cat','dipho_mass']# add CMS_hgg_mass



vhHadVars  = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar']


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
      trainFrames[proc]['proc'] = proc #not sure here
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


#some extra cuts that are applied for vhHad BDT in the AN
  trainTotal = trainTotal[trainTotal.dipho_leadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_subleadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_lead_ptoM>0.333]
  trainTotal = trainTotal[trainTotal.dipho_sublead_ptoM>0.25]
  #trainTotal = trainTotal[trainTotal.stage1cat>-1.] #fix later
  print 'done basic preselection cuts'


#add extra info to dataframe
  print 'about to add extra columns'
  #trainTotal['truthDipho'] = trainTotal.apply(truthDipho,axis=1) #will update def in addRowFunctions.py line 5
  #trainTotal['vhHadWeight'] = trainTotal.apply(vhHadWeight,axis=1)# needs to be imported
#  trainTotal['altDiphoWeight'] = trainTotal.apply(altDiphoWeight, axis=1)
  print 'all columns added'

#save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/trainTotal.pkl'%frameDir)
  print 'frame saved as %s/trainTotal.pkl'%frameDir
