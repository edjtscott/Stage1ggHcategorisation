#usual imports
import numpy as np
import pandas as pd
import xgboost as xg
import uproot as upr
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from os import path, system, listdir
from Tools.addRowFunctions import truthVBF, vbfWeight, cosThetaStar

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams', default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
parser.add_option('--useDataDriven', action='store_true', default=False, help='Use the data-driven replacement for backgrounds with non-prompt photons')
(opts,args)=parser.parse_args()

#setup global variables
trainDir = opts.trainDir
if trainDir.endswith('/'): trainDir = trainDir[:-1]
frameDir = trainDir.replace('trees','frames')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')

#define variables to be used
from Tools.variableDefinitions import allVarsGen, dijetVars, lumiDict

#possible to add new variables here - have done some suggested ones as an example
newVars = ['gghMVA_leadPhi','gghMVA_leadJEn','gghMVA_subleadPhi','gghMVA_SubleadJEn','gghMVA_SubsubleadJPt','gghMVA_SubsubleadJEn','gghMVA_subsubleadPhi','gghMVA_subsubleadEta']
newdiphoVars = ['dipho_leadIDMVA', 'dipho_subleadIDMVA','dipho_leadEta', 'dipho_subleadEta', 'dipho_cosphi', 'vtxprob', 'sigmarv', 'sigmawv']
allVarsGen += newVars
allVarsGen += newdiphoVars
dijetVars += newVars
dijetVars += newdiphoVars

#including the full selection
hdfQueryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>40.) and (dijet_SubJPt>30.) and (dijet_Mjj>250.)'
queryString = '(dipho_mass>100.) and (dipho_mass<180.) and (dipho_leadIDMVA>-0.2) and (dipho_subleadIDMVA>-0.2) and (dipho_lead_ptoM>0.333) and (dipho_sublead_ptoM>0.25) and (dijet_LeadJPt>40.) and (dijet_SubJPt>30.) and (dijet_Mjj>250.)'

if opts.useDataDriven:
  #define hdf input
  hdfDir = trainDir.replace('trees','hdfs')
  
  hdfList = []
  if hdfDir.count('all'):
    for year in lumiDict.keys():
      tempHdfFrame = pd.read_hdf('%s/VBF_with_DataDriven_%s_MERGEDFF_NORM_NEW.h5'%(hdfDir,year)).query(hdfQueryString)
      tempHdfFrame = tempHdfFrame[tempHdfFrame['sample']=='QCD']
      tempHdfFrame.loc[:, 'weight'] = tempHdfFrame['weight'] * lumiDict[year]
      hdfList.append(tempHdfFrame)
    hdfFrame = pd.concat(hdfList, sort=False)
  else:
    hdfFrame = pd.read_hdf('%s/VBF_with_DataDriven_%s_MERGEDFF_NORM_NEW.h5'%(hdfDir,hdfDir.split('/')[-2]) ).query(hdfQueryString)
    hdfFrame = hdfFrame[hdfFrame['sample']=='QCD']
  
  hdfFrame['proc'] = 'datadriven'

#define input files
procFileMap = {'ggh':'powheg_ggH.root', 'vbf':'powheg_VBF.root', 'vh':'powheg_VH.root',
               'dipho':'Dipho.root'}
theProcs = procFileMap.keys()
signals     = ['ggh','vbf','vh']
backgrounds = ['dipho']
if not opts.useDataDriven:
  procFileMap['gjet_anyfake'] = 'GJet.root'
  backgrounds.append('gjet_anyfake')

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainList = []
  #get trees from files, put them in data frames
  if not 'all' in trainDir:
    for proc,fn in procFileMap.iteritems():
      print 'reading in tree from file %s'%fn
      trainFile   = upr.open('%s/%s'%(trainDir,fn))
      if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
      elif proc in backgrounds: trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
      else: raise Exception('Error did not recognise process %s !'%proc)
      tempFrame = trainTree.pandas.df(allVarsGen).query(queryString)
      tempFrame['proc'] = proc
      trainList.append(tempFrame)
  else:
    for year in lumiDict.keys():
      for proc,fn in procFileMap.iteritems():
        thisDir = trainDir.replace('all',year)
        print 'reading in tree from file %s'%fn
        trainFile   = upr.open('%s/%s'%(thisDir,fn))
        if proc in signals: trainTree = trainFile['vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc]
        elif proc in backgrounds: trainTree = trainFile['vbfTagDumper/trees/%s_13TeV_GeneralDipho'%proc]
        else: raise Exception('Error did not recognise process %s !'%proc)
        tempFrame = trainTree.pandas.df(allVarsGen).query(queryString)
        tempFrame['proc'] = proc
        tempFrame.loc[:, 'weight'] = tempFrame['weight'] * lumiDict[year]
        trainList.append(tempFrame)
  print 'got trees and applied selections'

  #create one total frame
  if opts.useDataDriven: 
    trainList.append(hdfFrame)
  trainTotal = pd.concat(trainList, sort=False)
  del trainList
  del tempFrame
  if opts.useDataDriven: 
    del hdfFrame
  print 'created total frame'

  #add the target variable and the equalised weight
  trainTotal['truthVBF'] = trainTotal.apply(truthVBF,axis=1)
  trainTotal = trainTotal[trainTotal.truthVBF>-0.5]
  vbfSumW = np.sum(trainTotal[trainTotal.truthVBF==2]['weight'].values)
  gghSumW = np.sum(trainTotal[trainTotal.truthVBF==1]['weight'].values)
  bkgSumW = np.sum(trainTotal[trainTotal.truthVBF==0]['weight'].values)
  trainTotal['vbfWeight'] = trainTotal.apply(vbfWeight, axis=1, args=[vbfSumW,gghSumW,bkgSumW])
  trainTotal['dijet_centrality']=np.exp(-4.*((trainTotal.dijet_Zep/trainTotal.dijet_abs_dEta)**2))

  #save as a pickle file
  if not path.isdir(frameDir): 
    system('mkdir -p %s'%frameDir)
  trainTotal.to_pickle('%s/vbfDataDriven.pkl'%frameDir)
  print 'frame saved as %s/vbfDataDriven.pkl'%frameDir

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
vbfX  = trainTotal[dijetVars].values
vbfY  = trainTotal['truthVBF'].values
vbfTW = trainTotal['vbfWeight'].values
vbfFW = trainTotal['weight'].values
vbfM  = trainTotal['dipho_mass'].values

#do the shuffle
vbfX  = vbfX[theShuffle]
vbfY  = vbfY[theShuffle]
vbfTW = vbfTW[theShuffle]
vbfFW = vbfFW[theShuffle]
vbfM  = vbfM[theShuffle]

#split into train and test
vbfTrainX,  vbfTestX  = np.split( vbfX,  [trainLimit] )
vbfTrainY,  vbfTestY  = np.split( vbfY,  [trainLimit] )
vbfTrainTW, vbfTestTW = np.split( vbfTW, [trainLimit] )
vbfTrainFW, vbfTestFW = np.split( vbfFW, [trainLimit] )
vbfTrainM,  vbfTestM  = np.split( vbfM,  [trainLimit] )


#set up the training and testing matrices
trainMatrix = xg.DMatrix(vbfTrainX, label=vbfTrainY, weight=vbfTrainTW, feature_names=dijetVars)
testMatrix  = xg.DMatrix(vbfTestX, label=vbfTestY, weight=vbfTestFW, feature_names=dijetVars)
trainParams = {}
trainParams['objective'] = 'multi:softprob'
numClasses = 3
trainParams['num_class'] = numClasses
trainParams['nthread'] = 1
#trainParams['seed'] = 123456
#trainParams['max_depth'] = 7
#trainParams['n_estimators'] = 300
#trainParams['eta'] = 0.3
#trainParams['sub_sample'] = 0.9

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
print 'about to train the dijet BDT'
vbfModel = xg.train(trainParams, trainMatrix)
print 'done'

#save it
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
vbfModel.save_model('%s/vbfDataDriven%s.model'%(modelDir,paramExt))
print 'saved as %s/vbfDataDriven%s.model'%(modelDir,paramExt)

#evaluate performance using area under the ROC curve
vbfPredYtrain = vbfModel.predict(trainMatrix).reshape(vbfTrainY.shape[0],numClasses)
vbfPredYtest  = vbfModel.predict(testMatrix).reshape(vbfTestY.shape[0],numClasses)
vbfTruthYtrain = np.where(vbfTrainY==2, 0, 1)
vbfTruthYtest  = np.where(vbfTestY==2, 0, 1)
print 'Training performance:'
print 'area under roc curve for training set = %1.3f'%( 1.-roc_auc_score(vbfTruthYtrain, vbfPredYtrain[:,2], sample_weight=vbfTrainFW) )
print 'area under roc curve for test set     = %1.3f'%( 1.-roc_auc_score(vbfTruthYtest,  vbfPredYtest[:,2],  sample_weight=vbfTestFW)  )

#make some plots
#print (type(trainTotal[trainTotal.truthVBF==2]['dipho_lead_ptoM']))
#print(type(trainTotal[trainTotal.truthVBF==2]))

var_list = []
for (columnName, columnData) in trainTotal[trainTotal.truthVBF==2][dijetVars].iteritems():
    #print('column name', columnName)
    var_list.append(columnName)
print 'var_list', var_list

#x_label_list =[r'$p_^1/m_{\gamma\gamma}$',r'$p^2/m_{\gamma\gamma}$',r'$p_T^{j1}$',r'$p_T^{j2}$',r'$|\Delta\eta|$',r'$m_{jj}$',r'$C_{\gamma\gamma}$',r'$|\Delta\phi_{jj}|$',r'$\Delta R_{min}(\gamma,j)$',r'$|\Delta \phi_{\gamma\gamma,jj}|$']


plotDir = trainDir.replace('trees','plots')
#plotDir = '%s/%s'%(paramExt)
if not path.isdir(plotDir): 
  system('mkdir -p %s'%plotDir)

#roc_curve
fpr_tr, tpr_tr, thresholds_tr = roc_curve(vbfTruthYtrain, vbfPredYtrain[:,2], sample_weight=vbfTrainFW)
fpr_tst,tpr_tst, thresholds_tst = roc_curve(vbfTruthYtest,  vbfPredYtest[:,2],  sample_weight=vbfTestFW)

plt.figure(1)
#plt.plot(tpr_tr,fpr_tr,label = r'training set ROC curve (area = %1.3f $\pm$ 0.001 )'%( 1.-roc_auc_score(vbfTruthYtrain, vbfPredYtrain[:,2], sample_weight=vbfTrainFW)) )
#plt.plot(tpr_tst,fpr_tst,label = r'test set ROC curve (area = %1.3f $\pm$ 0.004)'%( 1.-roc_auc_score(vbfTruthYtest,  vbfPredYtest[:,2],  sample_weight=vbfTestFW))  )
plt.plot(tpr_tr,fpr_tr,label = r'training set ROC curve (area = 0.923 $\pm$ 0.001)')
plt.plot(tpr_tst,fpr_tst,label = r'test set ROC curve (area = 0.920 $\pm$ 0.004)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve with dipho variables')
plt.legend(loc='best',prop={'size': 12})
plt.savefig('%s/ROC_curve_wDipho.pdf'%plotDir)
print 'saved as %s/ROC_curve_wDipho.pdf'%plotDir




'''
for i in range(3,len(var_list)):
    plt.figure(i+1)
    plt.hist(trainTotal[trainTotal.truthVBF==2][var_list[i]],bins = 50,label = 'vbf',alpha = 0.5,normed = True)
    plt.hist(trainTotal[trainTotal.truthVBF==1][var_list[i]],bins = 50,label = 'ggh',alpha = 0.5,normed = True)
    plt.hist(trainTotal[trainTotal.truthVBF==0][var_list[i]],bins = 50,label = 'bkg',alpha = 0.5,normed = True)
    plt.xlabel(x_label_list[i], size = 14)
    plt.ylabel("events", size=14)
    plt.legend(loc='upper right')
    plt.savefig('%s/%s.pdf'%(plotDir,var_list[i]))
    print var_list[i]
    print 'saved as %s/%s.pdf'%(plotDir,var_list[i])


#the transverse momentum of the two leading photons, divided by the diphoton mass
plt.figure(1)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dipho_lead_ptoM'],range =[0.2,1.8],bins = 50,label = 'bkg',alpha = 0.5,normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dipho_lead_ptoM'],range =[0.2,1.8],bins = 50,label = 'ggh',alpha = 0.5,normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dipho_lead_ptoM'],range =[0.2,1.8],bins = 50,label = 'vbf',alpha = 0.5,normed = True)
plt.xlabel(r'$p^1/m_{\gamma\gamma}$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dipho_lead_ptoM.pdf'%plotDir)
print 'saved as %s/dipho_lead_ptoM.pdf'%plotDir

plt.figure(2)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dipho_sublead_ptoM'],range =[0.2,1.0],bins = 50,label = 'bkg',alpha = 0.5,normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dipho_sublead_ptoM'],range =[0.2,1.0],bins = 50,label = 'ggh',alpha = 0.5,normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dipho_sublead_ptoM'],range =[0.2,1.0],bins = 50,label = 'vbf',alpha = 0.5,normed = True)
plt.xlabel(r'$p^2/m_{\gamma\gamma}$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dipho_sublead_ptoM.pdf'%plotDir)
print 'saved as %s/dipho_sublead_ptoM.pdf'%plotDir

#the transverse momentum of the two leading jets
plt.figure(3)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dijet_LeadJPt'],range =[0.2,600],bins = 50,alpha = 0.5,label = 'bkg',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dijet_LeadJPt'],range =[0.2,600],bins = 50,alpha = 0.5,label = 'ggh',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dijet_LeadJPt'],range =[0.2,600],bins = 50,alpha = 0.5,label = 'vbf',normed = True)
plt.xlabel(r'$p^{j1}_T$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dijet_LeadJPt.pdf'%plotDir)
print 'saved as %s/dijet_LeadJPt.pdf'%plotDir

plt.figure(4)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dijet_SubJPt'],range =[0.2,600],bins = 50,alpha = 0.5,label = 'bkg',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dijet_SubJPt'],range =[0.2,600],bins = 50,alpha = 0.5,label = 'ggh',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dijet_SubJPt'],range =[0.2,600],bins = 50,alpha = 0.5,label = 'vbf',normed = True)
plt.xlabel(r'$p^{j2}_T$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dijet_SubJPt.pdf'%plotDir)
print 'saved as %s/dijet_SubJPt.pdf'%plotDir

#the dijet_abs_dEta
plt.figure(5)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dijet_abs_dEta'],range =[0,10],bins = 50,alpha = 0.5,label = 'bkg',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dijet_abs_dEta'],range =[0,10],bins = 50,alpha = 0.5,label = 'ggh',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dijet_abs_dEta'],range =[0,10],bins = 50,alpha = 0.5,label = 'vbf',normed = True)
plt.xlabel(r'$|\Delta\eta|$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dijet_abs_dEta.pdf'%plotDir)
print 'saved as %s/dijet_abs_dEta.pdf'%plotDir

#th dijet  invarian mass
plt.figure(6)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dijet_Mjj'],range =[0,6000],bins = 50,alpha = 0.5,label = 'bkg',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dijet_Mjj'],range =[0,6000],bins = 50,alpha = 0.5,label = 'ggh',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dijet_Mjj'],range =[0,6000],bins = 50,alpha = 0.5,label = 'vbf',normed = True)
plt.xlabel(r'$m_{jj}$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dijet_Mjj.pdf'%plotDir)
print 'saved as %s/dijet_Mjj.pdf'%plotDir

#dijet_centrality
plt.figure(7)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dijet_centrality'],range =[0,1.0],bins = 50,alpha = 0.5,label = 'bkg',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dijet_centrality'],range =[0,1.0],bins = 50,alpha = 0.5,label = 'ggh',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dijet_centrality'],range =[0,1.0],bins = 50,alpha = 0.5,label = 'vbf',normed = True)
plt.xlabel(r'$C_{\gamma\gamma}$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dijet_centrality.pdf'%plotDir)
print 'saved as %s/dijet_centrality.pdf'%plotDir

#dijet_dphi
plt.figure(8)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dijet_dphi'],range =[0,3.5],bins = 50,alpha = 0.5,label = 'bkg',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dijet_dphi'],range =[0,3.5],bins = 50,alpha = 0.5,label = 'ggh',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dijet_dphi'],range =[0,3.5],bins = 50,alpha = 0.5,label = 'vbf',normed = True)
plt.xlabel(r'$|\Delta\phi_{jj}|$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dijet_dphi.pdf'%plotDir)
print 'saved as %s/dijet_dphi.pdf'%plotDir

#dijet_minDRJetPho
plt.figure(9)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dijet_minDRJetPho'],range =[0,6],bins = 50,alpha = 0.5,label = 'bkg',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dijet_minDRJetPho'],range =[0,6],bins = 50,alpha = 0.5,label = 'ggh',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dijet_minDRJetPho'],range =[0,6],bins = 50,alpha = 0.5,label = 'vbf',normed = True)
plt.xlabel(r'$\Delta R_{min}(\gamma,j)$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dijet_minDRJetPho.pdf'%plotDir)
print 'saved as %s/dijet_minDRJetPho.pdf'%plotDir

#dijet_dipho_dphi_trunc
plt.figure(10)
plt.hist(trainTotal[trainTotal.truthVBF==0]['dijet_dipho_dphi_trunc'],range = [0,3.5],bins = 50,alpha = 0.5,label = 'bkg',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==1]['dijet_dipho_dphi_trunc'],range = [0,3.5],bins = 50,alpha = 0.5,label = 'ggh',normed = True)
plt.hist(trainTotal[trainTotal.truthVBF==2]['dijet_dipho_dphi_trunc'],range = [0,3.5],bins = 50,alpha = 0.5,label = 'vbf',normed = True)
plt.xlabel(r'$|\Delta \phi_{\gamma\gamma,jj}|$', size=14)
plt.ylabel("events", size=14)
plt.legend(loc='upper right')
plt.savefig('%s/dijet_dipho_dphi_trunc.pdf'%plotDir)
print 'saved as %s/dijet_dipho_dphi_trunc.pdf'%plotDir
'''


