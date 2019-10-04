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
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system

from addRowFunctions import truthProcess, ProcessWeight,truthProcessTwoClass,ProcessWeightTwoClass
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import fill_hist
import usefulStyle as useSty

from matplotlib import rc
from bayes_opt import BayesianOptimization


print 'imports done'

pd.options.mode.chained_assignment = None

np.random.seed(42)



#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-x','--modelDir', help = 'Directory for models')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams',default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
parser.add_option('-m','--modelName', default=None, help='Name of model for testing')
(opts,args)=parser.parse_args()


print 'option added'


#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]# slice the string to remove the last character i.e the "/"
frameDir = trainDir.replace('trees','frames')
modelDir = trainDir.replace('trees','models')

if opts.trainParams: opts.trainParams = opts.trainParams.split(',')#separate train options based on comma (used to define parameter pairs)

#get trees from files, put them in data frames
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'Dipho':'Dipho.root','GJet':'GJet.root','QCD':'QCD.root'}# a dictionary with file names
theProcs = procFileMap.keys()# list of keys i.e 'ggh','vbf','Data'



print 'processes defined'



allVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar', 'dipho_cosphi', 'vtxprob','sigmarv','sigmawv','weight','dipho_mass','dipho_leadEta','dipho_subleadEta','cos_dijet_dipho_dphi','dijet_Zep','dijet_jet1_QGL','dijet_jet2_QGL','dijet_dphi','dijet_minDRJetPho','dipho_leadPhi','dipho_subleadPhi','dipho_leadR9','dipho_subleadR9','dijet_dipho_dphi_trunc','dijet_dipho_pt','dijet_mva','dipho_dijet_MVA','dijet_jet1_pujid_mva','dijet_jet2_pujid_mva','dijet_jet1_RMS','dijet_jet2_RMS','dipho_lead_hoe','dipho_sublead_hoe','dipho_lead_elveto','dipho_sublead_elveto','jet1_HFHadronEnergyFraction','jet1_HFEMEnergyFraction', 'jet2_HFHadronEnergyFraction','jet2_HFEMEnergyFraction']


#allVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar', 'dipho_cosphi', 'vtxprob','sigmarv','sigmawv','weight', 'tempStage1bin','dipho_mass','dipho_leadEta','dipho_subleadEta','cos_dijet_dipho_dphi','dijet_Zep','dijet_jet1_QGL','dijet_jet2_QGL','dijet_dphi','dijet_minDRJetPho','dipho_leadPhi','dipho_subleadPhi','dipho_leadR9','dipho_subleadR9','dijet_dipho_dphi_trunc','dijet_dipho_pt','dijet_mva','dipho_dijet_MVA','dijet_jet1_pujid_mva','dijet_jet2_pujid_mva','dijet_jet1_RMS','dijet_jet2_RMS','dipho_lead_hoe','dipho_sublead_hoe','dipho_lead_elveto','dipho_sublead_elveto','jet1_HFHadronEnergyFraction','jet1_HFEMEnergyFraction', 'jet2_HFHadronEnergyFraction','jet2_HFEMEnergyFraction']



#'dipho_leadPhi','dipho_subleadPhi', 'dipho_leadR9','dipho_subleadR9','dipho_PToM','dijet_dipho_dphi_trunc','dijet_dipho_pt','dijet_dphi','dijet_mva','dipho_dijet_MVA','dijet_jet1_pujid_mva','dijet_jet2_pujid_mva','dijet_jet1_RMS','dijet_jet2_RMS']



print 'variables chosen'


#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:#if the dataframe option was not used while running, create dataframe from files in folder
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():#proc, fn are the pairs 'proc':'fn' in the file map 
      trainFile   = upr.open('%s/%s'%(trainDir,fn))
      print proc 
      print fn
  #is a reader and a writer of the ROOT file format using only Python and Numpy.
  #Unlike PyROOT and root_numpy, uproot does not depend on C++ ROOT. Instead, it uses Numpy to cast blocks of data from the ROOT file as Numpy arrays.
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
         trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc]
      print 'ok1'
      trainFrames[proc] = trainTree.pandas.df(allVars)
      print'ok'
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
  #trainTotal = trainTotal[((trainTotal['proc']=='ggh')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))|((trainTotal['proc']=='vbf')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))|((trainTotal['proc']=='Dipho')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))|((trainTotal['proc']=='GJet')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))|((trainTotal['proc']=='QCD')&(trainTotal['dipho_mass']>100.)&(trainTotal['dipho_mass']<180.))]# diphoton mass range
  trainTotal = trainTotal[trainTotal.dipho_mass<180.]# diphoton mass range
  trainTotal = trainTotal[trainTotal.dipho_mass>100.]
  print 'done mass cuts'
#some extra cuts that are applied for vhHad BDT in the AN
  trainTotal = trainTotal[trainTotal.dipho_leadIDMVA>-0.2]
  trainTotal = trainTotal[trainTotal.dipho_subleadIDMVA>-0.2]
  trainTotal = trainTotal[trainTotal.dipho_lead_ptoM>0.333]

  trainTotal = trainTotal[trainTotal.dipho_sublead_ptoM>0.25]
  print 'done basic preselection cuts'
#cut on the jet pT to require at least 2 jets
  trainTotal = trainTotal[trainTotal.dijet_LeadJPt>40.]
  trainTotal = trainTotal[trainTotal.dijet_SubJPt>30.]
  print 'done jet pT cuts'
#consider the VH hadronic mjj region (ideally to cut on gen mjj for this)
  trainTotal = trainTotal[trainTotal.dijet_Mjj>250.]
  print 'done mjj cuts'


#adding variables that need to be calculated

  trainTotal['dijet_dipho_dEta']=((trainTotal.dijet_leadEta+trainTotal.dijet_subleadEta)/2)-((trainTotal.dipho_leadEta+trainTotal.dipho_subleadEta)/2)
  trainTotal['dijet_centrality_gg']=np.exp(-4*(trainTotal.dijet_Zep/trainTotal.dijet_abs_dEta)**2)
  print 'calculated variables added'


  def adjust_qcd_weight(row):
      if row['proc']=='QCD':
         return row['weight']/25
      else:
         return row['weight']

  trainTotal['weightR'] = trainTotal.apply(adjust_qcd_weight, axis=1)

#add the target variable and the equalised weight
  trainTotal['truthProcess'] = trainTotal.apply(truthProcess,axis=1)#the truthProcess function returns 0 for ggh. 1 for vbf and 2 for background processes
  trainTotal['truthProcessTwoClass'] = trainTotal.apply(truthProcessTwoClass, axis=1)

  #gghSumW = np.sum(trainTotal[trainTotal.truthProcess==0]['weightR'].values)#summing weights of ggh events
  vbfSumW = np.sum(trainTotal[trainTotal.truthProcessTwoClass==1]['weightR'].values)#summing weights of vbf events
  dataSumW = np.sum(trainTotal[trainTotal.truthProcessTwoClass==0]['weightR'].values)#summing weights of data events  
  totalSumW = vbfSumW+dataSumW
  #getting number of events
  

  print 'weights before lum adjustment'
  print 'vbfSumW, bkgSumW, ratio_vbf_data = %.3f, %.3f, %.3f'%(vbfSumW,dataSumW, vbfSumW/dataSumW)
  print 'ratios'
  print 'vbf ratio, bkg ratio = %.3f, %.3f'%(vbfSumW/totalSumW, dataSumW/totalSumW)


  trainTotal['ProcessWeightTwoClass'] = trainTotal.apply(ProcessWeightTwoClass, axis=1, args=[dataSumW/vbfSumW])#multiply each of the VH weight values by sum of nonsig weight/sum of sig weight 

#applying lum factors for ggh and vbf for training without equalised weights
  def weight_adjust (row):
      if row['truthProcessTwoClass'] == 0:
         return 41500 * row['weightR']
      if row['truthProcessTwoClass'] == 1:
         return 41500 * row['weightR']
      if row['truthProcessTwoClass'] == 2:
         return 41500 * row ['weightR']

  trainTotal['weightLUM'] = trainTotal.apply(weight_adjust, axis=1)


  #gghSumW = np.sum(trainTotal[trainTotal.truthProcess==0]['weightLUM'].values)#summing weights of ggh events
  vbfSumW = np.sum(trainTotal[trainTotal.truthProcess==1]['weightLUM'].values)#summing weights of vbf events
  dataSumW = np.sum(trainTotal[trainTotal.truthProcess==0]['weightLUM'].values)#summing weights of data events
  totalSumW = vbfSumW+dataSumW

  print 'weights after lum adjustment'
  print 'vbfSumW, dataSumW, ratio_vbf_data = %.3f, %.3f, %.3f'%(vbfSumW,dataSumW,vbfSumW/dataSumW)
  print 'vbf ratio, bkg ratio = %.3f, %.3f'%(vbfSumW/totalSumW, dataSumW/totalSumW)


#trainTotal = trainTotal[trainTotal.truthVhHad>-0.5]

  print 'done weight equalisation'

#save as a pickle file
#if not path.isdir(frameDir): 
  system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/TwoClassTotal.pkl'%frameDir)
  print 'frame saved as %s/TwoClassTotal.pkl'%frameDir

#read in dataframe if above steps done before
else:

  trainTotal = pd.read_pickle('%s/%s'%(frameDir,opts.dataFrame))
print 'Successfully loaded the dataframe'


#______________________-getting the data.root for the bkg yields later__________________________
#dataTotal = None
#dataFile   = upr.open('%s/%s'%(trainDir,Data.root))
      
#dataTree = dataFile['vbfTagDumper/trees/Data_13TeV_GeneralDipho']
#dataFrame = trainTree.pandas.df(allVars)
#dataFrame['proc'] = 'Data'
#dataTotal = dataFrame
#dataTotal = dataTotal[dataTotal.dipho_mass>100.]
#dataTotal = dataTotal[dataTotal.dipho_mass<180.]# diphoton mass range*********remove the signal region?
  

#dataTotal = dataTotal[dataTotal.dipho_leadIDMVA>-0.2]
#dataTotal = dataTotal[dataTotal.dipho_subleadIDMVA>-0.2]
#dataTotal = dataTotal[dataTotal.dipho_lead_ptoM>0.333]
#dataTotal = dataTotal[dataTotal.dipho_sublead_ptoM>0.25]


#dataTotal = dataTotal[dataTotal.dijet_LeadJPt>40.]
#dataTotal = dataTotal[dataTotal.dijet_SubJPt>30.]
 
#dataTotal = dataTotal[dataTotal.dijet_Mjj>250.]



#____________________________________________
#__________________________________________________





#set up train set and randomise the inputs
trainFrac = 0.90


theShape = trainTotal.shape[0]#number of rows in total dataframe
theShuffle = np.random.permutation(theShape)
trainLimit = int(theShape*trainFrac)


#define the values needed for training as numpy arrays
#vhHadVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar','cos_dijet_dipho_dphi', 'dijet_dipho_dEta']#do not provide dipho_mass=>do not bias the BDT by the Higgs mass used in signal MC

#BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva','dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_nj', 'cosThetaStar', 'cos_dijet_dipho_dphi','dijet_dipho_dEta','dijet_centrality_gg','dijet_jet1_QGL','dijet_jet2_QGL','dijet_dphi','dijet_minDRJetPho']

#dijet MVA var
BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_centrality_gg','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc']

#dijet+diphoton MVA var
#BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_centrality_gg','dijet_dphi','dijet_minDRJetPho','dijet_dipho_dphi_trunc','dipho_leadEta','dipho_subleadEta','dipho_cosphi','dipho_leadIDMVA','dipho_subleadIDMVA','sigmarv','sigmawv','vtxprob']



#lessvar
#BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva','dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_nj', 'cosThetaStar', 'cos_dijet_dipho_dphi','dijet_dipho_dEta','dijet_centrality_gg','dijet_jet1_QGL','dijet_jet2_QGL','dijet_dphi','dijet_minDRJetPho','dipho_leadIDMVA','dipho_subleadIDMVA', 'dipho_cosphi','dipho_leadEta','dipho_subleadEta','dipho_leadPhi','dipho_subleadPhi','dijet_dipho_dphi_trunc','dijet_dipho_pt','dijet_mva','dipho_dijet_MVA']

#BDTVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva','dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_nj', 'cosThetaStar', 'cos_dijet_dipho_dphi','dijet_dipho_dEta','dijet_centrality_gg','dijet_jet1_QGL','dijet_jet2_QGL','dijet_dphi','dijet_minDRJetPho','dipho_leadIDMVA','dipho_subleadIDMVA', 'dipho_cosphi','vtxprob','sigmarv','sigmawv','dipho_leadEta','dipho_subleadEta','dipho_leadPhi','dipho_subleadPhi','dipho_leadR9','dipho_subleadR9','dijet_dipho_dphi_trunc','dijet_dipho_pt','dijet_mva','dipho_dijet_MVA','dijet_jet1_RMS','dijet_jet2_RMS','dipho_lead_hoe','dipho_sublead_hoe','dipho_lead_elveto','dipho_sublead_elveto','jet1_HFHadronEnergyFraction','jet1_HFEMEnergyFraction', 'jet2_HFHadronEnergyFraction','jet2_HFEMEnergyFraction']





BDTX  = trainTotal[BDTVars].values# the train input variables defined in the above list
BDTY  = trainTotal['truthProcessTwoClass'].values#the training target two classes 1 for vh had 0 for other processes 
BDTTW = trainTotal['ProcessWeightTwoClass'].values
BDTFW = trainTotal['weightLUM'].values
BDTM  = trainTotal['dipho_mass'].values


#do the shuffle
BDTX  = BDTX[theShuffle]
BDTY  = BDTY[theShuffle]
BDTTW = BDTTW[theShuffle]
BDTFW = BDTFW[theShuffle]
BDTM  = BDTM[theShuffle]

#split into train and test
BDTTrainX,  BDTTestX  = np.split( BDTX,  [trainLimit] )
BDTTrainY,  BDTTestY  = np.split( BDTY,  [trainLimit] )
BDTTrainTW, BDTTestTW = np.split( BDTTW, [trainLimit] )
BDTTrainFW, BDTTestFW = np.split( BDTFW, [trainLimit] )
BDTTrainM,  BDTTestM  = np.split( BDTM,  [trainLimit] )


#for v in trainTotal['truthProcessTwoClass']:
  #print v

print 'size of training labels'
print len(BDTTrainY)

print 'size of x train'
print len(BDTTrainX)
#set up the training and testing matrices
trainMatrix = xg.DMatrix(BDTTrainX, label=BDTTrainY, weight=BDTTrainTW, feature_names=BDTVars)
testMatrix  = xg.DMatrix(BDTTestX, label=BDTTestY, weight=BDTTestTW, feature_names=BDTVars)

#train on equalised weights
#trainMatrix = xg.DMatrix(BDTTrainX, label=BDTTrainY, weight=BDTTrainTW, feature_names=BDTVars)
#testMatrix  = xg.DMatrix(BDTTestX, label=BDTTestY, weight=BDTTestTW, feature_names=BDTVars)

trainParams = {}
#trainParams['objective'] = 'binary:logistic'
trainParams['objective'] = 'multi:softprob'
trainParams['num_class']=2

trainParams['nthread'] = 1#--number of parallel threads used to run xgboost



#playing with parameters
#trainParams['eta']=0.1
#trainParams['max_depth']=5
#trainParams['subsample']=0.5
#trainParams['colsample_bytree']=0.7
#trainParams['min_child_weight']=0
#trainParams['gamma']=0
trainParams['eval_metric']='merror'

trainParams['seed'] = 123456
#trainParams['reg_alpha']=0.1
#trainParams['reg_lambda']=

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

progress = dict()
watchlist  = [(trainMatrix,'train'), (testMatrix, 'eval')]


print 'trying cross-validation'
#x_parameters = {"max_depth":[5,6,7,8,10], "objective":'multi:softprob', "num_class":3, "nthread":1, "eta":[0.05,0.1,0.3], "subsample":[0.5, 0.8, 1.],"eval_metric":'merror', "seed":123456}
#xg.cv(x_parameters, trainMatrix)





#train the BDT (specify number of epochs here)
print 'about to train BDT'
TwoClassModel = xg.train(trainParams, trainMatrix,15,watchlist)
print 'done'
print progress

#ThreeClassModel = xg.Booster()
#ThreeClassModel.load_model('%s/%s'%(modelDir,opts.modelName))
 

#save it
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
TwoClassModel.save_model('%s/TwoClassModel%s.model'%(modelDir,paramExt))
print 'saved as %s/TwoClassModel%s.model'%(modelDir,paramExt)




#evaluate performance 
print 'predicting test and train sets from trained model'
BDTPredYtrain = TwoClassModel.predict(trainMatrix)
BDTPredYtest  = TwoClassModel.predict(testMatrix)


#print 'prediction probabilities column 0 -ggh'
#print BDTPredYtrain[:,0]
#print 'prediction probabilities column 1 -vbf'
#print BDTPredYtrain[:,1]
#print 'prediction probabilities column 2 -bkg'
#print BDTPredYtrain[:,2]

#print 'trying maximum value'
#print np.argmax(BDTPredYtrain,axis=1)


#print 'labels'


BDTPredClassTrain = np.argmax(BDTPredYtrain,axis=1)
BDTPredClassTest = np.argmax(BDTPredYtest,axis=1)
print 'testing argmax'
print np.argmax([0.01,0.015])

################################################################
print 'making MVA prob score plot for all'
plt.figure()
plt.title('MVA prob plot --trainset--vbf prob')
#plt.bins([0,1,2,3])

plt.hist((BDTPredYtrain[:,1])[(BDTTrainY==0)],bins=50,weights=BDTTrainFW[(BDTTrainY==0)], histtype='step',normed=1, color='blue',label='bkg', range = [0.,0.1])
#plt.hist((BDTPredYtrain[:,1])[(BDTTrainY==0)],bins=50,weights=BDTTrainFW[(BDTTrainY==0)], histtype='step',normed=1, color='red',label='ggh',range = [0.,1.])
plt.hist((BDTPredYtrain[:,1])[(BDTTrainY==1)],bins=50,weights=BDTTrainFW[(BDTTrainY==1)], histtype='step',normed=1, color='green',label='vbf',range = [0.,0.1])

plt.xlabel('BDT probabilities')

plt.legend()
plt.savefig('Two_MVA_prob_train_vbf_prob.png',bbox_inches = 'tight')
plt.savefig('Two_MVA_prob_train_vbf_prob.pdf',bbox_inches = 'tight')
###########################################################################
print 'making MVA prob score plot for all'
plt.figure()
plt.title('MVA prob plot --testset--vbf prob')
#plt.bins([0,1,2,3])

plt.hist((BDTPredYtest[:,1])[(BDTTestY==0)],weights=BDTTestFW[(BDTTestY==0)], histtype='step',normed=1, color='blue',label='bkg',range = [0.,0.1],bins=50)
#plt.hist((BDTPredYtest[:,1])[(BDTTestY==0)],weights=BDTTestFW[(BDTTestY==0)], histtype='step',normed=1, color='red',label='ggh',range = [0.,1.],bins=50)
plt.hist((BDTPredYtest[:,1])[(BDTTestY==1)],weights=BDTTestFW[(BDTTestY==1)], histtype='step',normed=1, color='green',label='vbf',range = [0.,0.1],bins=50)

plt.xlabel('BDT probabilities')

plt.legend()
plt.savefig('Two_MVA_prob_test_vbf_prob.png',bbox_inches = 'tight')
plt.savefig('Two_MVA_prob_test_vbf_prob.pdf',bbox_inches = 'tight')
################################################################################


################################################################################

print 'checking yields based on vbf probabilities for different probability cut values'

def computeBkg(hist, margin ):
    bkgVal = 9999.
    #if hist.GetEntries() > 0 and hist.Integral()>0.000001:
    if hist.GetEffectiveEntries() > 10 and hist.Integral()>0.000001:
      hist.Fit('expo')
      fit = hist.GetFunction('expo')
      bkgVal = fit.Integral(125. - margin, 125. + margin)
    return bkgVal


cutVal_list = [0.0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
testScale = 1./(1.-trainFrac)
for cutVal in cutVal_list:#BDT boundaries--we are using 3 VH tag categories?
  selectedvbf = testScale * np.sum( BDTTestFW * (BDTTestY==1) * ((BDTPredYtest[:,1])>cutVal) )
  #selectedggh = testScale * np.sum( BDTTestFW * (BDTTestY==0) * ((BDTPredYtest[:,1])>cutVal) )
  selectedbkg = testScale * np.sum( BDTTestFW * (BDTTestY==0) * ((BDTPredYtest[:,1])>cutVal) )

  selectedbkg_exp = BDTTestM[(BDTTestY==0)&((BDTPredYtest[:,1])>cutVal)]
  selectedbkg_exp_w = BDTTestFW[(BDTTestY==0)&((BDTPredYtest[:,1])>cutVal)]
  bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
  fill_hist(bkgHist, selectedbkg_exp, weights=selectedbkg_exp_w)
  background_per_GeV_from_exp = testScale*computeBkg(bkgHist,0.5)

  selectedbkg_perGeV = selectedbkg/70.
  selectedtotal = selectedvbf+selectedbkg 
  print 'Selected events for a cut value of %.2f: vbf %.3f,  bkg %.3f'%(cutVal, selectedvbf, selectedbkg)
  print 'fractions for a cut value of %.2f: vbf %.3f, bkg %.3f'%(cutVal,selectedvbf/selectedtotal, selectedbkg/selectedtotal) 
  print 'background per GeV is %.3f'%(selectedbkg_perGeV)
  print 'background per GeV from exp is %.3f'%(background_per_GeV_from_exp)

print 'checking the yields based on argmax-vbf class'
selectedvbf = testScale * np.sum( BDTTestFW * (BDTTestY==1) * (BDTPredClassTest==1))
#selectedggh = testScale * np.sum( BDTTestFW * (BDTTestY==0) * (BDTPredClassTest==1))
selectedbkg = testScale * np.sum( BDTTestFW * (BDTTestY==0) * (BDTPredClassTest==0))
selectedbkg_perGeV = selectedbkg/70.
selectedtotal = selectedvbf+selectedbkg
print 'Selected events with argmax: vbf %.3f, bkg %.3f'%(selectedvbf, selectedbkg)
print 'fractions for argmax selection: vbf %.3f, bkg %.3f'%(selectedvbf/selectedtotal, selectedbkg/selectedtotal)
print 'background per GeV is %.3f'%(selectedbkg_perGeV)



print 'checking the yields based on argmax-ggh class'
selectedvbf = testScale * np.sum( BDTTestFW * (BDTTestY==1) * (BDTPredClassTest==0))
#selectedggh = testScale * np.sum( BDTTestFW * (BDTTestY==0) * (BDTPredClassTest==0))
selectedbkg = testScale * np.sum( BDTTestFW * (BDTTestY==0) * (BDTPredClassTest==0))

selectedbkg_perGeV = selectedbkg/70.

selectedtotal = selectedvbf+selectedbkg
print 'Selected events with argmax: vbf %.3f, bkg %.3f'%(selectedvbf, selectedbkg)
print 'fractions for argmax selection: vbf %.3f, bkg %.3f'%(selectedvbf/selectedtotal, selectedbkg/selectedtotal)
print 'background per GeV is %.3f'%(selectedbkg_perGeV)


print 'checking the yields based on argmax-bkg class'
selectedvbf = testScale * np.sum( BDTTestFW * (BDTTestY==1) * (BDTPredClassTest==2))
#selectedggh = testScale * np.sum( BDTTestFW * (BDTTestY==0) * (BDTPredClassTest==2))
selectedbkg = testScale * np.sum( BDTTestFW * (BDTTestY==0) * (BDTPredClassTest==2))


selectedbkg_perGeV = selectedbkg/70.
selectedtotal = selectedvbf++selectedbkg
print 'Selected events with argmax: vbf %.3f, bkg %.3f'%(selectedvbf, selectedbkg)
print 'fractions for argmax selection: vbf %.3f, bkg %.3f'%(selectedvbf/selectedtotal, selectedbkg/selectedtotal)
print 'background per GeV is %.3f'%(selectedbkg_perGeV)


###############################################################################

#SCORE PLOT
print 'making MVA score plot'
plt.figure()
plt.title('MVA score plot --trainset')
#plt.bins([0,1,2,3])
#plt.hist(BDTPredClassTrain[(BDTTrainY==0)],bins=[0,1,2,3], weights=BDTTrainFW[(BDTTrainY==0)], histtype='step',normed=1, color='red',label='ggh')

plt.hist(BDTPredClassTrain[(BDTTrainY==1)],bins=[0,1,2,3], weights=BDTTrainFW[(BDTTrainY==1)], histtype='step',normed=1, color='green',label='vbf')
plt.hist(BDTPredClassTrain[(BDTTrainY==0)], bins=[0,1,2,3], weights=BDTTrainFW[(BDTTrainY==0)], histtype='step',normed=1, color='blue',label='bkg')

plt.xlabel('BDT class')

plt.legend()
plt.savefig('Two_MVA_score_train.png',bbox_inches = 'tight')
plt.savefig('Two_MVA_score_train.pdf',bbox_inches = 'tight')


plt.figure()
plt.title('MVA score plot --testset')
#plt.bins([0,1,2,3])
#plt.hist(BDTPredClassTest[(BDTTestY==0)], bins=[0,1,2,3],weights=BDTTestFW[(BDTTestY==0)],  histtype='step',normed=1, color='red',label='ggh')
plt.hist(BDTPredClassTest[(BDTTestY==1)], bins=[0,1,2,3],weights=BDTTestFW[(BDTTestY==1)],  histtype='step',normed=1, color='green',label='vbf')
plt.hist(BDTPredClassTest[(BDTTestY==0)], bins=[0,1,2,3],weights=BDTTestFW[(BDTTestY==0)],  histtype='step',normed=1, color='blue',label='bkg')


plt.xlabel('BDT score')
plt.legend()
plt.savefig('Two_MVA_score_test.png',bbox_inches = 'tight')
plt.savefig('Two_MVA_score_test.pdf',bbox_inches = 'tight')

print 'DONE'

from sklearn.metrics import roc_curve, auc, roc_auc_score

#plot roc curves

BDTTrainY_bkg_roc = np.where(BDTTrainY==0, BDTTrainY, 1)
BDTTestY_bkg_roc = np.where(BDTTestY==0, BDTTestY,1)

print 'Training performance:auc'
print 'area under roc curve for training set_bkg = %1.5f'%(1- roc_auc_score(BDTTrainY_bkg_roc, BDTPredYtrain[:,0], sample_weight = BDTTrainFW) )
print 'area under roc curve for test set_bkg     = %1.5f'%(1- roc_auc_score(BDTTestY_bkg_roc, BDTPredYtest[:,0], sample_weight = BDTTestFW)  )
roc_auc_train_bkg = 1- roc_auc_score(BDTTrainY_bkg_roc, BDTPredYtrain[:,0], sample_weight = BDTTrainFW)
roc_auc_test_bkg = 1- roc_auc_score(BDTTestY_bkg_roc, BDTPredYtest[:,0], sample_weight = BDTTestFW) 

fpr, tpr, thresholds= roc_curve(BDTTrainY_bkg_roc, BDTPredYtrain[:,0], pos_label=0, sample_weight = BDTTrainFW)
#fpr, tpr, thresholds = roc_curve(BDTTrainY, BDTPredYtrain, pos_label=0)
#roc_auc=auc(fpr,tpr)
print 'loaded train roc curve'
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve train (area =%0.2f )'%roc_auc_train_bkg)
#plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve train')
fpr, tpr, thresholds = roc_curve(BDTTestY_bkg_roc, BDTPredYtest[:,0], pos_label=0,sample_weight = BDTTestFW)
#roc_auc=auc(fpr,tpr)
print 'loaded test roc curve'

plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve test (area =%0.2f )'%roc_auc_test_bkg)
#plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve test')
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck', zorder=5)
plt.legend()
plt.savefig('Two_ROC_bkg.png',bbox_inches = 'tight')
plt.savefig('Two_ROC_bkg.pdf',bbox_inches = 'tight')


#plot roc curves
BDTTrainY_vbf_roc = np.where(BDTTrainY==1, BDTTrainY, 0)
BDTTestY_vbf_roc = np.where(BDTTestY==1, BDTTestY,0)

print 'Training performance:auc'
print 'area under roc curve for training set_vbf = %1.5f'%(roc_auc_score(BDTTrainY_vbf_roc, BDTPredYtrain[:,1], sample_weight = BDTTrainFW) )
print 'area under roc curve for test set_vbf     = %1.5f'%(roc_auc_score(BDTTestY_vbf_roc, BDTPredYtest[:,1], sample_weight = BDTTestFW)  )

roc_auc_train_vbf = roc_auc_score(BDTTrainY_vbf_roc, BDTPredYtrain[:,1], sample_weight = BDTTrainFW)
roc_auc_test_vbf = roc_auc_score(BDTTestY_vbf_roc, BDTPredYtest[:,1], sample_weight = BDTTestFW) 


fpr, tpr, thresholds = roc_curve(BDTTrainY_vbf_roc, BDTPredYtrain[:,1], pos_label=1,sample_weight = BDTTrainFW)
#roc_auc=auc(fpr,tpr)
print 'loaded train roc curve'
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve train (area =%0.2f )'%roc_auc_train_vbf)
#plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve train')

fpr, tpr, thresholds = roc_curve(BDTTestY_vbf_roc, BDTPredYtest[:,1], pos_label=1,sample_weight = BDTTestFW)
#roc_auc=auc(fpr,tpr)
print 'loaded test roc curve'
plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve test (area =%0.2f)'%roc_auc_test_vbf)

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck', zorder=5)
plt.legend()
plt.savefig('Two_ROC_vbf.png',bbox_inches = 'tight')
plt.savefig('Two_ROC_vbf.pdf',bbox_inches = 'tight')


#plot roc curves
#BDTTrainY_bkg_roc = np.where(BDTTrainY=2, BDTTrainY, 0)
#BDTTestY_bkg_roc = np.where(BDTTestY==2, BDTTestY,0)

#BDTTrainY_bkg_roc_auc = np.where(BDTTrainY_bkg_roc==0,BDTTrainY_bkg_roc,1)
#BDTTestY_bkg_roc_auc = np.where(BDTTestY_bkg_roc==0,BDTTestY_bkg_roc ,1)

#print 'Training performance:auc'
#print 'area under roc curve for training set_bkg = %1.5f'%(roc_auc_score(BDTTrainY_bkg_roc_auc, BDTPredYtrain[:,2], sample_weight = BDTTrainFW) )
#print 'area under roc curve for test set_bkg     = %1.5f'%(roc_auc_score(BDTTestY_bkg_roc_auc, BDTPredYtest[:,2],sample_weight = BDTTestFW)  )

#roc_auc_train_bkg = roc_auc_score(BDTTrainY_bkg_roc_auc, BDTPredYtrain[:,2], sample_weight = BDTTrainFW)
#roc_auc_test_bkg = roc_auc_score(BDTTestY_bkg_roc_auc, BDTPredYtest[:,2],sample_weight = BDTTestFW)


#fpr, tpr, thresholds = roc_curve(BDTTrainY_bkg_roc, BDTPredYtrain[:,2], pos_label=2,sample_weight = BDTTrainFW)
#roc_auc=auc(fpr,tpr)
#print 'loaded train roc curve'
#plt.figure()
#plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve train (area =%0.2f )'%roc_auc_train_bkg)
#plt.plot(fpr, tpr, color='blue', lw=2,label='ROC curve train')

#fpr, tpr, thresholds = roc_curve(BDTTestY_bkg_roc, BDTPredYtest[:,2], pos_label=2,sample_weight = BDTTestFW)
#roc_auc=auc(fpr,tpr)
#print 'loaded test roc curve'

#plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve test (area =%0.2f )'%roc_auc_test_bkg)
#plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve test')
#plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck', zorder=5)
#plt.legend()
#plt.savefig('ROC_bkg.png',bbox_inches = 'tight')
#plt.savefig('ROC_bkg.pdf',bbox_inches = 'tight')

#train test comparison

print 'checking for overtraining'
plt.figure()
plt.title('Train/Test comparison')


#plt.hist(BDTPredClassTrain[(BDTTrainY==0)],bins=[0,1,2,3], weights=BDTTrainFW[(BDTTrainY==0)], histtype='step',normed=1, color='red',label='ggh (train)')
plt.hist(BDTPredClassTrain[(BDTTrainY==1)],bins=[0,1,2,3], weights=BDTTrainFW[(BDTTrainY==1)], histtype='step',normed=1, color='green',label='vbf (train')
plt.hist(BDTPredClassTrain[(BDTTrainY==0)], bins=[0,1,2,3], weights=BDTTrainFW[(BDTTrainY==0)], histtype='step',normed=1, color='blue',label='bkg (train')

plt.xlabel('BDT class')


decisions = []
weight    = []

d1 = BDTPredClassTest[(BDTTestY==0)]
d2 = BDTPredClassTest[(BDTTestY==1)]
#d3 = BDTPredClassTest[(BDTTestY==2)]

w1 = BDTTestFW[(BDTTestY==0)]
w2 = BDTTestFW[(BDTTestY==1)]
#w3 = BDTTestFW[(BDTTestY==2)]

decisions += [d1, d2]
weight    += [w1, w2]

low  = min(np.min(d) for d in decisions)
high = max(np.max(d) for d in decisions)
low_high = (low,high)

hist, bins = np.histogram(decisions[0],bins=[0,1,2,3], range=low_high, normed=True, weights = weight[0] )

scale = len(decisions[0]) / sum(hist)
err = np.sqrt(hist * scale) / scale
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

plt.errorbar(center+0.5, hist, yerr=err, fmt='.', c='r', label='bkg (test)', markersize=8,capthick=0)


hist, bins = np.histogram(decisions[1],bins=bins, range=low_high, normed=True, weights = weight[1])
scale = len(decisions[1]) / sum(hist)
err = np.sqrt(hist * scale) / scale

plt.errorbar(center+0.5, hist, yerr=err, fmt='.', c='g', label='vbf (test)', markersize=8,capthick=0)


#hist, bins = np.histogram(decisions[2],bins=bins, range=low_high, normed=True, weights = weight[2])
#scale = len(decisions[2]) / sum(hist)
#err = np.sqrt(hist * scale) / scale

#plt.errorbar(center+0.5, hist, yerr=err, fmt='.', c='b', label='bkg (test)', markersize=8,capthick=0)


plt.legend()
plt.savefig('TwoClass_train_test_comp.png',bbox_inches = 'tight')
plt.savefig('TwoClass_train_test_comp.pdf',bbox_inches = 'tight')

#test the importance of the features
#plt.figure(figsize=(20,40))
plt.figure()
matplotlib.rc('ytick', labelsize=4) 
#grid = np.random.random((10,10))
ax=xg.plot_importance(TwoClassModel,show_values=False)
#ax.imshow(grid, extent=[0,100000,0,10000], aspect=600)
ax.plot()
plt.show
plt.savefig('TwoClass_featureImportance.png',bbox_inches = 'tight')
plt.savefig('TwoClass_featureImportance.pdf',bbox_inches = 'tight')

print 'done feature importance'


#PLOTTING INTERESTING VARIABLES

plotVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva','dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_nj', 'cosThetaStar','dipho_mass', 'cos_dijet_dipho_dphi','dijet_dipho_dEta','dijet_centrality_gg','dijet_jet1_QGL','dijet_jet2_QGL','dijet_dphi','dijet_minDRJetPho','dipho_leadIDMVA','dipho_subleadIDMVA', 'dipho_cosphi','vtxprob','sigmarv','sigmawv','dipho_leadEta','dipho_subleadEta','dipho_leadPhi','dipho_subleadPhi','dipho_leadR9','dipho_subleadR9','dijet_dipho_dphi_trunc','dijet_dipho_pt','dijet_mva','dipho_dijet_MVA','dijet_jet1_RMS','dijet_jet2_RMS','dipho_lead_hoe','dipho_sublead_hoe','dipho_lead_elveto','dipho_sublead_elveto','jet1_HFHadronEnergyFraction','jet1_HFEMEnergyFraction', 'jet2_HFHadronEnergyFraction','jet2_HFEMEnergyFraction']


plotVarsX=['lead photon pT/mgg', 'sublead photon pT/mgg', 'diphoton MVA score', 'lead jet eta', 'sublead jet eta', 'lead jet pT', 'sublead jet pT', 'dijet dEta', 'dijet Mjj', 'number of jets', 'oosThetaStar','diphoton invariant mass','dijet dipho cos phi', 'dijet dipho dEta','dijet centrality gg','dijet jet1 QGL','dijet jet2 QGL','dijet dphi','dijet minDRJetPho','dipho leadIDMVA','dipho subleadIDMVA', 'dipho cosphi','vtxprob','sigmarv','sigmawv','dipho leadEta','dipho subleadEta','dipho leadPhi','dipho subleadPhi','dipho leadR9','dipho subleadR9','dijet dipho dphi trunc','dijet dipho pt','dijet mva','dipho dijet MVA','dijet jet1 RMS','dijet jet2 RMS','dipho lead hoe','dipho sublead hoe','dipho lead elveto','dipho sublead elveto','jet1 HFHadronEnergyFraction','jet1 HFEMEnergyFraction', 'jet2 HFHadronEnergyFraction','jet2 HFEMEnergyFraction']

plotVarsR=[(0,5),(0,5),(-1,1), (-3,3),(-3,3),(0,300),(0,300),(0,6),(0,4000),(0,8),(-1,1),(100,180),(-1,1),(-3,3),(0,600),(0,8),(0,1),(-100, 100),(-10,10),(-1,1),(-1,1),(0,1),(-1,1),(0,1),(0,1.1),(0,0.1),(0,0.1),(-3,3),(-3,3),(-3.2,3.2),(0,2),(0,2),(-10,10),(0,400),(-1,1),(-1,1),(-10,10),(-10,10),(0,0.04),(0,0.04),(0.95,1.5),(0.95,1.5),(-1,1),(-1,1),(-1,1),(-1,1)]

#separate dataframes to plot
df_ggh = trainTotal[trainTotal['proc']=='ggh']
df_VBF = trainTotal[trainTotal['proc']=='vbf']
df_Data = trainTotal[trainTotal['proc']=='Data']


numpy_ggh_weight = df_ggh['weight'].values
numpy_VBF_weight = df_VBF['weight'].values
numpy_Data_weight = df_Data['weight'].values


#defining plot function

def plot_variable(var='cosThetaStar', var_label = '$\cos\,\theta^*$', setrange=(-1,1)):


  numpy_ggh = df_ggh[var].values
  numpy_VBF = df_VBF[var].values
  numpy_Data = df_Data[var].values
  

  plt.figure(figsize=(6,6))
  plt.rc('text', usetex=True)


  plt.title(r'\textbf{CMS}\,\textit{Preliminary Simulation}',loc='left')

  plt.hist(numpy_ggh, bins=50,weights=numpy_ggh_weight,histtype='step', normed=1, color = 'red',range=setrange, label = 'ggh',linewidth=2.0)
  plt.hist(numpy_VBF, bins=50,weights=numpy_VBF_weight,histtype='step', normed=1, color = 'green', range=setrange, label = 'vbf',linewidth=2.0)
  plt.hist(numpy_Data, bins=50,weights=numpy_Data_weight,histtype='step', normed=1, color = 'blue', range=setrange,label = 'bkg',linewidth=2.0)

  plt.legend(loc='best')
  plt.xlabel(var_label)
  plt.ylabel('1/N dN/d(%s)'%var_label)
  plt.savefig('ThreeClass_var_plots/%s.png'%var,bbox_inches = 'tight')
  plt.savefig('ThreeClass_var_plots/%s.pdf'%var,bbox_inches = 'tight')



var_list = range(0,len(plotVars))

print 'plotting relevant variables'
#for i in var_list:
  #plot_variable(var=plotVars[i], var_label =plotVarsX[i], setrange=plotVarsR[i])

print 'all plots created'
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

