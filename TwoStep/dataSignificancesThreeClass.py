#ort ROOT as r
import numpy as np
import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system
import uproot as upr
from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthProcess
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty
from catOptim import CatOptim

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Name of dataframe if it already exists')
parser.add_option('-s','--signalFrame', default=None, help='Name of signal dataframe if it already exists')
parser.add_option('-m','--diphomodelName', default=None, help='Name of diphomodel for testing')
parser.add_option('-v','--dijetmodelName', default = None, help = 'Name of dijet model for testing')
parser.add_option('-n','--nIterations', default=10000, help='Number of iterations to run for random significance optimisation')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models').replace('ForVBF/','')
trainFrac = 0.9
validFrac = 0.1


#define the different sets of variables used                                                                                                                                                                      
diphoVars  = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'dipho_cosphi','vtxprob','sigmarv','sigmawv']

dijetVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_centrality_gg','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc']



allVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'dipho_cosphi','vtxprob','sigmarv','sigmawv','HTXSstage1cat','dipho_mass','weight','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc','dijet_Zep','dipho_dijet_ptHjj']

dataVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'dipho_cosphi','vtxprob','sigmarv','sigmawv','dipho_mass','weight','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc','dijet_Zep','dipho_dijet_ptHjj']


#load train file
print 'loading train file'
trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))#called trainTotal.pkl for now, obtained after running diphotonCategorisation for now
print 'adding truth info'
trainTotal['truthDipho'] = trainTotal.apply(truthDipho,axis=1)
trainTotal['truthProcess'] = trainTotal.apply(truthProcess,axis=1)#the truthProcess function returns 0 for ggh. 1 for vbf and 2 for background processes

trainTotal['dijet_centrality_gg']=np.exp(-4*(trainTotal.dijet_Zep/trainTotal.dijet_abs_dEta)**2)
#trainTotal['weight_LUM'] = 41.5*(trainTotal.weight)
trainTotal['weight_LUM'] = (trainTotal.weight)


dataFileMap = {'Data':'Data.root'}# a dictionary with file names                                                                        
#theProcs = procFileMap.keys()# list of keys i.e 'ggh','vbf','Data'   

print 'making data frame'
dataTotal = None
#if not opts.dataFrame:
dataFrames = {}
  #get the trees, turn them into arrays
for proc,fn in dataFileMap.iteritems():
    trainFile   = upr.open('%s/%s'%(trainDir,fn))
    if (proc=='Dipho'):
         trainTree = trainFile['vbfTagDumper/trees/dipho_13TeV_GeneralDipho']
    elif (proc=='GJet'):
         trainTree = trainFile['vbfTagDumper/trees/gjet_anyfake_13TeV_GeneralDipho']
    elif (proc=='QCD'):
         trainTree = trainFile['vbfTagDumper/trees/qcd_anyfake_13TeV_GeneralDipho']
    elif (proc=='ggh'):
         trainTree = trainFile['vbfTagDumper/trees/ggh_125_13TeV_GeneralDipho']
    elif (proc=='vbf'):
         trainTree = trainFile['vbfTagDumper/trees/vbf_125_13TeV_GeneralDipho']
    else:
         trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
    #trainFile   = r.TFile('%s/%s'%(trainDir,fn))
    #if proc[-1].count('h') or 'vbf' in proc: trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_VBFDiJet'%proc)
    #else: trainTree = trainFile.Get('vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc)
#    trainTree.SetBranchStatus('nvtx',0)
 #   trainTree.SetBranchStatus('dijet_*',0)
  #  trainTree.SetBranchStatus('dijet_Mjj',1)
   # trainTree.SetBranchStatus('dijet_LeadJPt',1)
   # trainTree.SetBranchStatus('dijet_SubJPt',1)
   # trainTree.SetBranchStatus('dZ',0)
   # trainTree.SetBranchStatus('centralObjectWeight',0)
   # trainTree.SetBranchStatus('rho',0)
   # trainTree.SetBranchStatus('nvtx',0)
   # trainTree.SetBranchStatus('event',0)
   # trainTree.SetBranchStatus('lumi',0)
   # trainTree.SetBranchStatus('processIndex',0)
   # trainTree.SetBranchStatus('run',0)
   # trainTree.SetBranchStatus('npu',0)
   # trainTree.SetBranchStatus('puweight',0)
    #newFile = r.TFile('/vols/cms/es811/Stage1categorisation/trainTrees/new.root','RECREATE')
    #newTree = trainTree.CloneTree()
    #dataFrames[proc] = pd.DataFrame( tree2array(newTree) )
    #del newTree
    #del newFile
    dataFrames[proc] = trainTree.pandas.df(dataVars)
    dataFrames[proc]['proc'] = proc
print 'got trees'

dataTotal = dataFrames['Data']
  
#then filter out the events into only those with the phase space we are interested in
dataTotal = dataTotal[dataTotal.dipho_mass>100.]
dataTotal = dataTotal[dataTotal.dipho_mass<180.]
print 'done mass cuts'
  
  #apply the full VBF preselection
dataTotal = dataTotal[dataTotal.dipho_leadIDMVA>-0.2]
dataTotal = dataTotal[dataTotal.dipho_subleadIDMVA>-0.2]
dataTotal = dataTotal[dataTotal.dipho_lead_ptoM>0.333]
dataTotal = dataTotal[dataTotal.dipho_sublead_ptoM>0.25]
dataTotal = dataTotal[dataTotal.dijet_Mjj>250.]
dataTotal = dataTotal[dataTotal.dijet_LeadJPt>40.]
dataTotal = dataTotal[dataTotal.dijet_SubJPt>30.]
print 'done VBF preselection cuts'
 # dataTotal['dijet_centrality_gg']=np.exp(-4*(dataTotal.dijet_Zep/dataTotal.dijet_abs_dEta)**2)
  #save as a pickle file
  #if not path.isdir(frameDir): 
    #system('mkdir -p %s'%frameDir)
  #dataTotal.to_pickle('%s/dataTotal.pkl'%frameDir)
  #print 'frame saved as %s/dataTotal.pkl'%frameDir
#else:
  #dataTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))

dataTotal['dijet_centrality_gg']=np.exp(-4*(dataTotal.dijet_Zep/dataTotal.dijet_abs_dEta)**2)




#define the variables used as input to the classifier
trainM  = trainTotal['dipho_mass'].values
trainFW = trainTotal['weight_LUM'].values
#trainP  = trainTotal['truthDipho'].values
#diphoV  = trainTotal['VBFMVAValue'].values
trainH  = trainTotal['dipho_dijet_ptHjj'].values
trainJ  = trainTotal['dijet_Mjj'].values
trainL  = trainTotal['dijet_LeadJPt'].values

dataM  = dataTotal['dipho_mass'].values
dataFW = np.ones(dataM.shape[0])
#dataV  = dataTotal['VBFMVAValue'].values
dataH  = dataTotal['dipho_dijet_ptHjj'].values
dataJ  = dataTotal['dijet_Mjj'].values
dataL  = dataTotal['dijet_LeadJPt'].values

# either use what is already packaged up or perform inference
#if not opts.modelName:
  #diphoD  = trainTotal['diphomvaxgb'].values
  #dataD  = dataTotal['diphomvaxgb'].values
#else:

#obtain diphoton MVA predictions
diphoX = trainTotal[diphoVars].values
data_diphoX  = dataTotal[diphoVars].values
diphoP = trainTotal['truthDipho'].values
diphoMatrix = xg.DMatrix(diphoX, label=diphoP, weight=trainFW, feature_names=diphoVars)
datadiphoMatrix  = xg.DMatrix(data_diphoX,  label=dataFW, weight=dataFW,  feature_names=diphoVars)
diphoModel = xg.Booster()
diphoModel.load_model('%s/%s'%(modelDir,opts.diphomodelName))
diphoMVA = diphoModel.predict(diphoMatrix)
data_diphoMVA  = diphoModel.predict(datadiphoMatrix)

#obtain dijet MVA predictions
dijetX = trainTotal[dijetVars].values
data_dijetX = dataTotal[dijetVars].values
dijetP = trainTotal['truthProcess'].values
dijetMatrix = xg.DMatrix(dijetX,label = dijetP, weight = trainFW, feature_names = dijetVars)
datadijetMatrix = xg.DMatrix(data_dijetX , label = dataFW, weight = dataFW, feature_names = dijetVars)##################
dijetModel = xg.Booster()
dijetModel.load_model('%s/%s'%(modelDir,opts.dijetmodelName))
dijetMVA = dijetModel.predict(dijetMatrix)

data_dijetMVA = dijetModel.predict(datadijetMatrix)


dijetMVA_ggHprob = 1-dijetMVA[:,0]
dijetMVA_vbfprob = dijetMVA[:,1]
data_dijetMVA_ggHprob =1-data_dijetMVA[:,0]
data_dijetMVA_vbfprob = data_dijetMVA[:,1]

x_ggh = (dijetMVA[:,0])[dijetP==0]
y_ggh =(dijetMVA[:,1])[dijetP==0]
w_ggh = trainFW[dijetP==0]

#print x_ggh
plt.hist2d(x_ggh, y_ggh,bins = 50,weights = w_ggh, range = [[0,1],[0,1]], label = 'ggh events')
plt.title('ggH events')
plt.xlabel('ggh probability')
plt.ylabel('vbf probability')
plt.savefig('2D_ggh_sig.png',bbox_inches = 'tight')
plt.savefig('2D_ggh_sig.pdf',bbox_inches = 'tight')


#now estimate significance using the amount of background in a plus/mins 1 sigma window
#set up parameters for the optimiser
ptHjjCut = 25.
ranges = [ [0,1.], [0,1.],[0,1]]
#ranges = [[0,1],[0,1]]
names  = ['DijetBDTvbf','DijetBDTggh','DiphotonBDT']
#names = ['ggh prob', 'diphoton bdt']

printStr = ''

#plotDir = trainDir.replace('trees','plots')
#if not path.isdir(plotDir): 
 # system('mkdir -p %s'%plotDir)
#plotDir = '%s/%s/significanceThreeClass'%(plotDir,opts.modelName.replace('.model',''))
#if not path.isdir(plotDir): 
 # system('mkdir -p %s'%plotDir)

#optimising ot three categories (stage 0)
sigWeights = trainFW * (dijetP==1) * (trainJ>400)*(trainL<200)
bkgWeights = dataFW 
optimiser = CatOptim(sigWeights, trainM, [dijetMVA_vbfprob,dijetMVA_ggHprob, diphoMVA], bkgWeights, dataM, [data_dijetMVA_vbfprob, data_dijetMVA_ggHprob, data_diphoMVA], 1, ranges, names)
#optimiser = CatOptim(sigWeights, trainM, [dijetMVA_ggHprob, diphoMVA], bkgWeights, dataM, [data_dijetMVA_ggHprob, data_diphoMVA], 3, ranges, names)
#optimiser.setConstantBkg(True)
print 'about to optimise cuts'

optimiser.optimise(opts.intLumi, opts.nIterations)
printStr += 'Results for 3 vbf categories'
printStr += optimiser.getPrintableResult()

print printStr
