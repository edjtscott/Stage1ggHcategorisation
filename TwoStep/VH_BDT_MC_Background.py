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
procFileMap = {'ggh':'ggH.root', 'vbf':'VBF.root', 'vh':'VH.root', 'Dipho':'Dipho.root','GJet':'GJet.root','QCD':'QCD.root'}# a dictionary with file names
theProcs = procFileMap.keys()# list of keys i.e 'ggh','vbf','vh'

allVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar', 'dipho_cosphi', 'vtxprob','sigmarv','sigmawv','weight','dipho_mass','dipho_leadEta','dipho_subleadEta','cos_dijet_dipho_dphi','dijet_Zep','dijet_jet1_QGL','dijet_jet2_QGL','dijet_minDRJetPho']


###____adding data file for the HIG 16 040 comparison________
#dataFileMap = {'Data':'Data.root'}
#dataTotal = None
#if not opts.dataFrame:
  #dataFrames = {}
  #get the trees, turn them into arrays
  #for proc,fn in dataFileMap.iteritems():
    #dataFile = upr.open('%s/%s'%(trainDir,fn))
    #dataTree = dataFile['vbfTagDumper/trees/Data_13TeV_GeneralDipho']
    #dataFrames[proc] = dataTree.pandas.df(allVars)
    #dataFrames[proc]['proc'] = proc
  #print 'got trees for normal background'

  #dataTotal = dataFrames['Data']

  #dataTotal = dataTotal[dataTotal.dipho_mass>100.]
  #dataTotal = dataTotal[dataTotal.dipho_mass<180.]

  #dataTotal = dataTotal[dataTotal.dipho_leadIDMVA>-0.9]
  #dataTotal = dataTotal[dataTotal.dipho_subleadIDMVA>-0.9]
  #dataTotal = dataTotal[dataTotal.dijet_Mjj>60.]
  #dataTotal = dataTotal[dataTotal.dijet_Mjj<120.]

  #dataTotal = dataTotal[dataTotal.dipho_lead_ptoM>0.5]
  #dataTotal = dataTotal[dataTotal.dipho_sublead_ptoM>0.25]
  #dataTotal = dataTotal[dataTotal.dipho_mva>0.79]
  #dataTotal = dataTotal[dataTotal.dijet_nj>1]
  #dataTotal = dataTotal[dataTotal.dijet_LeadJPt>40]
  ##dataTotal = dataTotal[dataTotal.dijet_SubJPt>40]
  #dataTotal = dataTotal[dataTotal.dijet_leadEta<2.4]
  #dataTotal = dataTotal[dataTotal.dijet_subleadEta<2.4]
  #dataTotal = dataTotal[dataTotal.dijet_minDRJetPho>0.4]
  #dataTotal = dataTotal[dataTotal.cosThetaStar<0.5]

#_______________________________________________



allVars = ['dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar', 'dipho_cosphi', 'vtxprob','sigmarv','sigmawv','weight', 'dipho_mass','dipho_leadEta','dipho_subleadEta','cos_dijet_dipho_dphi','dijet_Zep','dijet_jet1_QGL','dijet_jet2_QGL','dijet_minDRJetPho']


#either get existing data frame or create it
trainTotal = None
df_H1G = None
if not opts.dataFrame:#if the dataframe option was not used while running, create dataframe from files in folder
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():#proc, fn are the pairs 'proc':'fn' in the file map 
      trainFile   = upr.open('%s/%s'%(trainDir,fn))
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
      #trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_VBFDiJet'%proc]
      trainFrames[proc] = trainTree.pandas.df(allVars)
      trainFrames[proc]['proc'] = proc #adding a column for the process
  print 'got trees'

#create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  df_HIG = pd.concat(trainList)
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

#__________________________________HIG_16_040 cuts for comparison____________________________

  print 'checking HIG 16 040 significance'  

  df_HIG = df_HIG[df_HIG.dipho_mass>100.]
  df_HIG = df_HIG[df_HIG.dipho_mass<180.]
  
  df_HIG = df_HIG[df_HIG.dipho_leadIDMVA>-0.9]
  df_HIG = df_HIG[df_HIG.dipho_subleadIDMVA>-0.9]
  df_HIG = df_HIG[df_HIG.dijet_Mjj>60.]
  df_HIG = df_HIG[df_HIG.dijet_Mjj<120.]

  df_H1G = df_HIG[df_HIG.dipho_lead_ptoM>0.5]
  df_HIG = df_HIG[df_HIG.dipho_sublead_ptoM>0.25]
  df_HIG = df_HIG[df_HIG.dipho_mva>0.79]
  df_HIG = df_HIG[df_HIG.dijet_nj>1]
  df_HIG = df_HIG[df_HIG.dijet_LeadJPt>40]
  df_HIG = df_HIG[df_HIG.dijet_SubJPt>40]
  df_HIG = df_HIG[df_HIG.dijet_leadEta<2.4]
  df_HIG = df_HIG[df_HIG.dijet_subleadEta<2.4]
  df_HIG = df_HIG[df_HIG.dijet_minDRJetPho>0.4]
  df_HIG = df_HIG[df_HIG.cosThetaStar<0.5]
  

  df_HIG['truthVhHad'] = df_HIG.apply(truthVhHad,axis=1)
  signal_weight = np.sum(df_HIG[df_HIG.truthVhHad==1]['weight'].values)
  ggh_background_weight = np.sum(df_HIG[df_HIG.truthVhHad==0]['weight'].values) 

  #____INTEGRATING BACKGROUND UNDER THE PEAK_____
  print 'finding background under peak'
  #bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
  #fill_hist(bkgHist, dataTotal['dipho_mass'].values, weights=dataTotal['weight'].values)
   
  #sigHist = r.TH1F('sigHistTemp', 'sigHistTemp', 160,100,180)
  #fill_hist(sigHist, df_HIG['dipho_mass'].values, weights = df_HIG['weight'].values)


  #print 'data weights'
  #print dataTotal['weight'].values
  #print 'getting effective sigma'
 # from catOptim import getRealSigma
  #def getRealSigma(hist):
   # sigma = 2.
    #if hist.GetEntries() > 0 and hist.Integral()>0.000001:
     # hist.Fit('gaus')
      #fit = hist.GetFunction('gaus')
      #sigma = fit.GetParameter(2)
    #return sigma

  #effective_sigma = getRealSigma(sigHist)
  #print 'effective sigma is'
  #print effective_sigma

  #from catOptim import computeBkg
  #def computeBkg(hist, effSigma ):
    #bkgVal = 9999.
    #if hist.GetEntries() > 0 and hist.Integral()>0.000001:
    #if hist.GetEffectiveEntries() > 10 and hist.Integral()>0.000001:
      #hist.Fit('expo')
      #fit = hist.GetFunction('expo')
      #bkgVal = fit.Integral(125. - effSigma, 125. + effSigma)
    #return bkgVal
  
  #background_under_peak = computeBkg(bkgHist,effective_sigma)
  #print 'background under peak is'
  #print background_under_peak  
  
  #background_weight = np.sum(dataTotal['weight'].values) 
  #background_weight = background_under_peak
  #berr =3
  #term_a = (signal_weight*41.5)+background_weight+berr
  #term_b = (1+np.true_divide(signal_weight*41.5,background_weight+berr))
  #c_bin = np.sqrt(2*(term_a*np.log(term_b)-signal_weight*41.5))

  #print 'the significance by applying HIG 16 040 cuts is'
  #print c_bin

#____________________________________________________________________________________________




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
  trainTotal['truthVhHad'] = trainTotal.apply(truthVhHad,axis=1)#the truthVhHad function returns 1 if the gen HTXS stage 0 category of the event id VH hadronic
  sigSumW = np.sum(trainTotal[trainTotal.truthVhHad==1]['weightR'].values)#summing weights of vh hadronic events (selected by gen HTXS category)
  bkgSumW = np.sum(trainTotal[trainTotal.truthVhHad==0]['weightR'].values)#summing weights of non-vh hadronic events (selected by gen HTXS stage 0 category
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

vhHadVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva','dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta','dijet_Mjj','dijet_nj', 'cosThetaStar', 'cos_dijet_dipho_dphi','dijet_dipho_dEta','dijet_centrality_gg','dijet_jet1_QGL','dijet_jet2_QGL']



vhHadX  = trainTotal[vhHadVars].values# the train input variables defined in the above list
vhHadY  = trainTotal['truthVhHad'].values#the training target two classes 1 for vh had 0 for other processes 
vhHadTW = trainTotal['vhHadWeight'].values
vhHadFW = trainTotal['weightR'].values
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
trainParams['nthread'] = 1#--number of parallel threads used to run xgboost
#playing with parameters

trainParams['max_depth']=6
trainParams['subsample']=1
trainParams['colsample_bytree']=1
trainParams['min_child_weight']= 0
trainParams['gamma']=0
trainParams['eval_metric']='auc'

trainParams['seed'] = 123456
#trainParams['reg_alpha']=
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

#train the BDT (specify number of epochs here)
print 'about to train diphoton BDT'
vhHadModel = xg.train(trainParams, trainMatrix,25,watchlist)
print 'done'
#print progress



#save it
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
vhHadModel.save_model('%s/vhHadModel%s.model'%(modelDir,paramExt))
print 'saved as %s/vhHadModel%s.model'%(modelDir,paramExt)




#evaluate performance 
print 'predicting test and train sets from trained model'
vhHadPredYtrain = vhHadModel.predict(trainMatrix)
vhHadPredYtest  = vhHadModel.predict(testMatrix)



#SCORE PLOT
print 'making MVA score plot'
plt.figure()
plt.title('VH MVA score plot --trainset')
plt.hist(vhHadPredYtrain[(vhHadTrainY>0.5)], bins=50,weights=vhHadTrainTW[(vhHadTrainY>0.5)],range=[0,1], alpha=0.5, histtype='stepfilled',normed=1, color='red',label='VH')
plt.hist(vhHadPredYtrain[(vhHadTrainY<0.5)], bins=50,weights=vhHadTrainTW[(vhHadTrainY<0.5)],range=[0,1], alpha=0.5, histtype='stepfilled',normed=1, color='green',label='ggH')
plt.xlabel('VH BDT score')

plt.legend()
plt.savefig('MVA_score_train.png',bbox_inches = 'tight')
plt.savefig('MVA_score_train.pdf',bbox_inches = 'tight')


plt.figure()
plt.title('VH MVA score plot --testset')
plt.hist(vhHadPredYtest[(vhHadTestY>0.5)], bins=50,weights=vhHadTestFW[(vhHadTestY>0.5)],range=[0,1], alpha=0.5, histtype='stepfilled',normed=1, color='red',label='VH')
plt.hist(vhHadPredYtest[(vhHadTestY<0.5)], bins=50,weights=vhHadTestFW[(vhHadTestY<0.5)],range=[0,1], alpha=0.5, histtype='stepfilled',normed=1, color='green',label='ggH')
plt.xlabel('VH BDT score')
plt.legend()
plt.savefig('MVA_score_test.png',bbox_inches = 'tight')
plt.savefig('MVA_score_test.pdf',bbox_inches = 'tight')


#train test comparison

print 'checking for overtraining'
plt.figure()
plt.title('Train/Test comparison')
plt.hist(vhHadPredYtrain[(vhHadTrainY>0.5)], bins=50,weights=vhHadTrainTW[(vhHadTrainY>0.5)],range=[0,1], alpha=0.5, histtype='stepfilled',normed=1, color='red',label='S (train)')
plt.hist(vhHadPredYtrain[(vhHadTrainY<0.5)], bins=50,weights=vhHadTrainTW[(vhHadTrainY<0.5)],range=[0,1], alpha=0.5, histtype='stepfilled',normed=1, color='green',label='B (train)')
plt.xlabel('VH BDT score')


decisions = []
weight    = []
d1 = vhHadPredYtest[(vhHadTestY>0.5)]
d2 = vhHadPredYtest[(vhHadTestY<0.5)]
w1 = vhHadTestFW[(vhHadTestY>0.5)]
w2 = vhHadTestFW[(vhHadTestY<0.5)]
decisions += [d1, d2]
weight    += [w1, w2]

low  = min(np.min(d) for d in decisions)
high = max(np.max(d) for d in decisions)
low_high = (low,high)

hist, bins = np.histogram(decisions[0],bins=50, range=low_high, normed=True, weights = weight[0] )

scale = len(decisions[0]) / sum(hist)
err = np.sqrt(hist * scale) / scale
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

plt.errorbar(center, hist, yerr=err, fmt='.', c='r', label='S (test)', markersize=8,capthick=0)


hist, bins = np.histogram(decisions[1],bins=bins, range=low_high, normed=True, weights = weight[1])
scale = len(decisions[1]) / sum(hist)
err = np.sqrt(hist * scale) / scale

plt.errorbar(center, hist, yerr=err, fmt='.', c='b', label='B (test)', markersize=8,capthick=0)

plt.legend()
plt.savefig('train_test_comp.png',bbox_inches = 'tight')
plt.savefig('train_test_comp.pdf',bbox_inches = 'tight')


# train performance
print 'Training performance:auc'
print 'area under roc curve for training set = %1.5f'%( roc_auc_score(vhHadTrainY, vhHadPredYtrain, sample_weight=vhHadTrainFW) )
print 'area under roc curve for test set     = %1.5f'%( roc_auc_score(vhHadTestY,  vhHadPredYtest,  sample_weight=vhHadTestFW)  )

from sklearn.metrics import roc_curve, auc
#plot roc curves
fpr, tpr, thresholds = roc_curve(vhHadTrainY, vhHadPredYtrain, pos_label=1)
roc_auc=auc(fpr,tpr)
print 'loaded train roc curve'
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve train (area =%0.2f )'%roc_auc)


fpr, tpr, thresholds = roc_curve(vhHadTestY, vhHadPredYtest, pos_label=1)
roc_auc=auc(fpr,tpr)
print 'loaded test roc curve'

plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve test (area =%0.2f )'%roc_auc)

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck', zorder=5)
plt.legend()
plt.savefig('ROC.png',bbox_inches = 'tight')
plt.savefig('ROC.pdf',bbox_inches = 'tight')


#test the importance of the features
plt.figure(figsize=(6,6))
plt.figure()
ax=xg.plot_importance(vhHadModel,show_values=False)
ax.plot()
plt.show
plt.savefig('featureImportance.png',bbox_inches = 'tight')
plt.savefig('featureImportance.pdf',bbox_inches = 'tight')

print 'done feature importance'

#check yields for various working points
testScale = 1./(1.-trainFrac)
for cutVal in np.arange(0.1,0.91,0.05):#BDT boundaries--we are using 3 VH tag categories?
  selectedSig = opts.intLumi * testScale * np.sum( vhHadTestFW * (vhHadTestY==1) * (vhHadPredYtest>cutVal) )
  selectedBkg = opts.intLumi * testScale * np.sum( vhHadTestFW * (vhHadTestY==0) * (vhHadPredYtest>cutVal) )
  print 'Selected events for a cut value of %.2f: S %.3f, B %.3f'%(cutVal, selectedSig, selectedBkg)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#PLOTTING INTERESTING VARIABLES

plotVars = ['dipho_lead_ptoM','dipho_sublead_ptoM','dipho_mva', 'dijet_leadEta','dijet_subleadEta','dijet_LeadJPt','dijet_SubJPt','dijet_abs_dEta', 'dijet_Mjj', 'dijet_nj', 'cosThetaStar','dipho_mass','cos_dijet_dipho_dphi', 'dijet_dipho_dEta','dijet_centrality_gg','dijet_jet1_QGL', 'dijet_jet2_QGL']

#latex
#plotVarsX = ['$p_{T}^{\gamma_{1}}\m_{\gamma\gamma}$', '$p_{T}^{\gamma_{2}}\m_{\gamma\gamma}$', 'Diphoton MVA score', '$\eta_{j_1}$','$\eta_{j_2}$','$p_T^{j_1}$','$p_T^{j_2}$', '$\mid\Delta\eta_jjj}\mid$', '$m_{jj}$','number of jets', '$\cos\theta^*$', '$m_{\gamma\gamma}$']

plotVarsX=['lead photon pT/mgg', 'sublead photon pT/mgg', 'diphoton MVA score', 'lead jet eta', 'sublead jet eta', 'lead jet pT', 'sublead jet pT', 'dijet dEta', 'dijet Mjj', 'number of jets', 'oosThetaStar','diphoton invariant mass','dijet dipho cos phi', 'dijet dipho dEta','dijet centrality gg','dijet jet1 QGL','dijet jet2 QGL']

plotVarsR=[(0,5),(0,5),(-1,1), (-3,3),(-3,3),(0,300),(0,300),(0,6),(60,120),(0,8),(-1,1),(100,180),(-1,1),(-3,3),(0,600),(0,8),(0,1),(-100, 100),(-100,100)]

#separate dataframes to plot
df_ggh = trainTotal[trainTotal['proc']=='ggh']
df_VBF = trainTotal[trainTotal['proc']=='vbf']
df_VH = trainTotal[trainTotal['proc']=='vh']
df_EWqqH = trainTotal[(trainTotal['proc']=='vbf')|(trainTotal['proc']=='vh')]
df_dipho = trainTotal[(trainTotal['proc']=='Dipho')]
df_gjet = trainTotal[(trainTotal['proc']=='GJet')]
df_qcd = trainTotal[(trainTotal['proc']=='QCD')]



print 'ggh sum weight'
print np.sum(df_ggh['weight'].values)

numpy_ggh_weight = df_ggh['weight'].values
numpy_VBF_weight = df_VBF['weight'].values
numpy_VH_weight = df_VH['weight'].values
numpy_EWqqH_weight = df_EWqqH['weight'].values
numpy_dipho_weight = df_dipho['weight'].values
numpy_gjet_weight = df_gjet['weight'].values
numpy_qcd_weight = df_qcd['weight'].values


#defining plot function

def plot_variable(var='cosThetaStar', var_label = '$\cos\,\theta^*$', setrange=(-1,1) , plot_type = 'EW qqH vs ggH'):
    
   
  numpy_ggh = df_ggh[var].values
  numpy_VBF = df_VBF[var].values
  numpy_VH = df_VH[var].values
  numpy_EWqqH = df_EWqqH[var].values
  numpy_dipho = df_dipho[var].values
  numpy_gjet = df_gjet[var].values
  numpy_qcd = df_qcd[var].values

  plt.figure(figsize=(6,6))
  plt.rc('text', usetex=True)


  plt.title(r'\textbf{CMS}\,\textit{Preliminary Simulation}',loc='left')
  
  if (plot_type=='EW qqH vs ggH'):
      plt.hist(numpy_ggh, bins=50,weights=numpy_ggh_weight,histtype='step', normed=1, color = 'green',range=setrange, label = 'ggh',linewidth=2.0)
      plt.hist(numpy_EWqqH, bins=50,weights=numpy_EWqqH_weight,histtype='step', normed=1, color = 'red',range=setrange, label = 'VH',linewidth=2.0)
      

  if (plot_type=='separate VH and VBF'): 
     plt.hist(numpy_ggh, bins=50,weights=numpy_ggh_weight,histtype='step', normed=1, color = 'green',range=setrange, label = 'ggh',linewidth=2.0)
     plt.hist(numpy_VBF, bins=50,weights=numpy_VBF_weight,histtype='step', normed=1, color = 'blue', range=setrange, label = 'vbf',linewidth=2.0)
     plt.hist(numpy_VH, bins=50,weights=numpy_VH_weight,histtype='step', normed=1, color = 'red', range=setrange,label = 'vh',linewidth=2.0)

  if (plot_type=='EWqqH vs ggH vs SM-sep'):
      plt.hist(numpy_ggh, bins=50,weights=numpy_ggh_weight,histtype='step', normed=1, color = 'green',range=setrange, label = 'ggh',linewidth=2.0)
      plt.hist(numpy_EWqqH, bins=50,weights=numpy_EWqqH_weight,histtype='step', normed=1, color = 'red',range=setrange, label = 'VH',linewidth=2.0)
      plt.hist(numpy_dipho, bins=50,weights=numpy_dipho_weight,histtype='step', normed=1, color = 'blue',range=setrange, label = 'dipho',linewidth=2.0)
      plt.hist(numpy_gjet, bins=50,weights=numpy_gjet_weight,histtype='step', normed=1, color = 'orange',range=setrange, label = 'gjet',linewidth=2.0)
      plt.hist(numpy_qcd, bins=50,weights=numpy_qcd_weight,histtype='step', normed=1, color = 'magenta',range=setrange, label = 'qcd',linewidth=2.0)



  plt.legend(loc='best')
  plt.xlabel(var_label)
  plt.ylabel('1/N dN/d(%s)'%var_label)
  plt.savefig('var_plots/%s.png'%var,bbox_inches = 'tight') 
  plt.savefig('var_plots/%s.pdf'%var,bbox_inches = 'tight')



var_list = range(0,len(plotVars))

print 'plotting relevant variables'
for i in var_list:
  plot_variable(var=plotVars[i], var_label =plotVarsX[i], setrange=plotVarsR[i], plot_type = 'EWqqH vs ggH vs SM-sep')

print 'all plots created'
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


