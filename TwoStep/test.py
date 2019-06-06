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

from matplotlib import rc
pd.options.mode.chained_assignment = None


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



#plotting

plotVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar','dipho_mass']

#latex
#plotVarsX = ['$p_{T}^{\gamma_{1}}\m_{\gamma\gamma}$', '$p_{T}^{\gamma_{2}}\m_{\gamma\gamma}$', 'Diphoton MVA score', '$\eta_{j_1}$','$\eta_{j_2}$','$p_T^{j_1}$','$p_T^{j_2}$', '$\mid\Delta\eta_{jj}\mid$', '$m_{jj}$','number of jets', '$\cos\theta^*$', '$m_{\gamma\gamma}$'] 

plotVarsX=['lead photon pT', 'sublead photon pT', 'diphoton MVA score', 'lead jet eta', 'sublead jet eta', 'lead jet pT', 'sublead jet pT', 'dijet dEta', 'dijet Mjj', 'number of jets', 'cosThetaStar','diphoton invariant mass']

plotVarsR=[(0,5),(0,5),(-1,1), (-3,3),(-3,3),(0,800),(0,800),(0,6),(0,1500),(0,8),(-1,1),(100,180)]




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
  trainTotal = trainTotal[trainTotal.dipho_mass>100.]
  trainTotal = trainTotal[trainTotal.dipho_mass<180.]
  print 'done mass cuts'


#some extra cuts that are applied for vhHad BDT in the AN
  trainTotal = trainTotal[trainTotal.dipho_leadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_subleadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_lead_ptoM>0.333]
  trainTotal = trainTotal[trainTotal.dipho_sublead_ptoM>0.25]
#  trainTotal = trainTotal[trainTotal.stage1cat>-1.] #fix later
  print 'done basic preselection cuts'


#plots

#df_ggh = trainTotal[trainTotal['proc']=='ggh']
#df_VBF = trainTotal[trainTotal['proc']=='vbf']
#df_VH = trainTotal[trainTotal['proc']=='vh']


#sample plot

#numpy_ggh = df_ggh['cosThetaStar'].values
#numpy_VBF = df_VBF['cosThetaStar'].values
#numpy_VH = df_VH['cosThetaStar'].values

#numpy_ggh_weight = df_ggh['weight'].values
#numpy_VBF_weight = df_VBF['weight'].values
#numpy_VH_weight = df_VH['weight'].values
#plt.figure(figsize=(6,6))
#plt.rc('text', usetex=True)
##plt.rc('font', family='serif')

#plt.title(r'\textbf{CMS}\,\textit{Preliminary Simulation}',loc='left')
#plt.hist(numpy_ggh, bins=50,weights=numpy_ggh_weight,histtype='step', normed=1, color = 'green',range=(-1,1), label = 'ggh',linewidth=2.0)
#plt.hist(numpy_VBF, bins=50,weights=numpy_VBF_weight,histtype='step', normed=1, color = 'blue', range=(-1,1), label = 'vbf',linewidth=2.0)
#plt.hist(numpy_VH, bins=50,weights=numpy_VH_weight,histtype='step', normed=1, color = 'red', range=(-1,1),label = 'vh',linewidth=2.0)
#plt.legend(loc='upper center')


##plt.xticks([100,200,300,400,500,600,700,800])
#plt.xlabel(r'$\cos\theta^*$')
#plt.ylabel(r'1/N dN/d($\cos\theta^*$)')
##plt.xlim([40,800])
##plt.ylim([0,0.01])
##plt.savefig('cosThetaStar.png',bbox_inches = 'tight')
##plt.savefig('cosThetaStar.pdf',bbox_inches = 'tight')



#defining plot function

df_ggh = trainTotal[trainTotal['proc']=='ggh']
df_VBF = trainTotal[trainTotal['proc']=='vbf']
df_VH = trainTotal[trainTotal['proc']=='vh']
numpy_ggh_weight = df_ggh['weight'].values
numpy_VBF_weight = df_VBF['weight'].values
numpy_VH_weight = df_VH['weight'].values



def plot_variable(var='cosThetaStar', var_label = '$\cos\,\theta^*$', setrange=(-1,1)):
    
   
  numpy_ggh = df_ggh[var].values
  numpy_VBF = df_VBF[var].values
  numpy_VH = df_VH[var].values

  plt.figure(figsize=(6,6))
  plt.rc('text', usetex=True)


  plt.title(r'\textbf{CMS}\,\textit{Preliminary Simulation}',loc='left')
  plt.hist(numpy_ggh, bins=50,weights=numpy_ggh_weight,histtype='step', normed=1, color = 'green',range=setrange, label = 'ggh',linewidth=2.0)
  plt.hist(numpy_VBF, bins=50,weights=numpy_VBF_weight,histtype='step', normed=1, color = 'blue', range=setrange, label = 'vbf',linewidth=2.0)
  plt.hist(numpy_VH, bins=50,weights=numpy_VH_weight,histtype='step', normed=1, color = 'red', range=setrange,label = 'vh',linewidth=2.0)

  plt.legend(loc='upper center')
  plt.xlabel(var_label)
  plt.ylabel('1/N dN/d(%s)'%var_label)
  plt.savefig('var_plots/%s.png'%var,bbox_inches = 'tight') 
  plt.savefig('var_plots/%s.pdf'%var,bbox_inches = 'tight')


var_list = range(0,12)


for i in var_list:
  plot_variable(var=plotVars[i], var_label =plotVarsX[i], setrange=plotVarsR[i])

