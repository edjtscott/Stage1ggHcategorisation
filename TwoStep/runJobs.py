from os import system, path, getcwd
from otherHelpers import submitJob

dryRun = False
#dryRun = True

myDir = getcwd()
baseDir = '/vols/cms/es811/Stage1categorisation'
#years = ['2016','2017']
years = ['2016']
intLumi=None

#script    = 'diphotonCategorisation.py'
#paramSets = [None,'max_depth:10']
#models    = None
#dataFrame = 'trainTotal.pkl'
#sigFrame  = None

#script    = 'dataSignificances.py'
#models    = ['altDiphoModel.model','diphoModel.model']
#paramSets = None
#dataFrame = 'dataTotal.pkl'
#sigFrame  = 'signifTotal.pkl'

script    = 'dataMCcheckSidebands.py'
models    = ['altDiphoModel.model','diphoModel.model']
paramSets = None
dataFrame = 'dataTotal.pkl'
sigFrame  = 'trainTotal.pkl'

if __name__=='__main__':
  for year in years:
    jobDir = '%s/Jobs/%s/%s' % (myDir, script.replace('.py',''), year)
    if not path.isdir( jobDir ): system('mkdir -p %s'%jobDir)
    trainDir  = '%s/%s/trees'%(baseDir,year)
    theCmd = 'python %s -t %s '%(script, trainDir)
    if dataFrame: 
      theCmd += '-d %s '%dataFrame
    if sigFrame: 
      theCmd += '-s %s '%sigFrame
    if intLumi: 
      theCmd += '--intLumi %s '%intLumi
    if paramSets and models:
      exit('ERROR do not expect both parameter set options and models. Exiting..')
    elif paramSets: 
      for params in paramSets:
        fullCmd = theCmd + '--trainParams %s '%params
        submitJob( jobDir, fullCmd, params=params, dryRun=dryRun )
    elif models:
      for model in models:
        fullCmd = theCmd + '-m %s '%model
        submitJob( jobDir, fullCmd, model=model, dryRun=dryRun )
