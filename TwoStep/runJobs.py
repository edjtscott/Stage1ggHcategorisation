from os import system, path, getcwd

myDir = getcwd()

baseDir = '/vols/cms/es811/Stage1categorisation'

script   = 'diphotonCategorisation.py'

#years = ['2016','2017']
years = ['2016']

paramSets = [None,'max_depth:10']

#dataFrame = None
dataFrame = 'trainTotal.pkl'

dryRun = False
#dryRun = True

def submitJob( jobDir, theCmd, params=None ):
  outName = '%s/sub__'%jobDir
  if params: 
    params = params.split(',')
    for pair in params:
      pair = pair.split(':')
      outName += '%s_%s__'%(pair[0],pair[1])
    outName = outName[:-2]
    outName += '.sh'
  else:
    outName += 'None.sh'
  with open('submitTemplate.sh') as inFile:
    with open(outName,'w') as outFile:
      for line in inFile.readlines():
        if '!CMD!' in line:
          line = line.replace('!CMD!','"%s"'%theCmd)
        elif '!MYDIR!' in line:
          line = line.replace('!MYDIR!',myDir)
        elif '!NAME!' in line:
          line = line.replace('!NAME!',outName.replace('.sh',''))
        outFile.write(line)
  subCmd = 'qsub -q hep.q -o %s -e %s -l h_vmem=24G %s' %(outName.replace('.sh','.log'), outName.replace('.sh','.err'), outName) 
  print subCmd
  if not dryRun:
    system(subCmd)

if __name__=='__main__':
  for year in years:
    jobDir = '%s/Jobs/%s/%s' % (myDir, script.replace('.py',''), year)
    if not path.isdir( jobDir ): system('mkdir -p %s'%jobDir)
    trainDir  = '%s/%s/trees'%(baseDir,year)
    theCmd = 'python %s -t %s '%(script, trainDir)
    if dataFrame: 
      dataFrame = '%s/%s'%(trainDir.replace('trees','frames'), dataFrame)
      theCmd += '-d %s '%dataFrame
    for params in paramSets:
      submitJob( jobDir, theCmd, params )
