from collections import OrderedDict as od
import numpy as np
import ROOT as r
r.gROOT.SetBatch(True)
from root_numpy import fill_hist
import usefulStyle as useSty


class Bests:
  '''Class to store and update best values during a category optimisation'''

  def __init__(self, nCats):
    self.nCats = nCats

    self.totSignif = -999.
    self.sigs      = [-999. for i in range(self.nCats)]
    self.bkgs      = [-999. for i in range(self.nCats)]
    self.nons      = [-999. for i in range(self.nCats)]
    self.signifs   = [-999. for i in range(self.nCats)]

  def update(self, sigs, bkgs, nons):
    signifs = []
    totSignifSq = 0.
    for i in range(self.nCats):
      sig = sigs[i] 
      bkg = bkgs[i] 
      non = nons[i] 
      signif = self.getAMS(sig, bkg+non)
      #signif = self.getAMS(sig, bkg+(2.*non)) ## experimental higher penalty for ggH
      #signif = self.getAMS(sig, bkg+(non*non)) ## experimental higher penalty for ggH
      signifs.append(signif)
      totSignifSq += signif*signif
    totSignif = np.sqrt( totSignifSq )
    if totSignif > self.totSignif:
      self.totSignif = totSignif
      for i in range(self.nCats):
        self.sigs[i]     = sigs[i]
        self.bkgs[i]     = bkgs[i]
        self.nons[i]     = nons[i]
        self.signifs[i]  = signifs[i]
      return True
    else:
      return False

  def getAMS(self, s, b, breg=3.):
    b = b + breg
    val = 0.
    if b > 0.:
      val = (s + b)*np.log(1. + (s/b))
      val = 2*(val - s)
      val = np.sqrt(val)
    return val

  def getSigs(self):
    return self.sigs

  def getBkgs(self):
    return self.bkgs

  def getSignifs(self):
    return self.signifs

  def getTotSignif(self):
    return self.totSignif


class CatOptim:
  '''
  Class to run category optimisation via random search for arbitrary numbers of categories and input discriminator distributions
                            _
                           | \
                           | |
                           | |
      |\                   | |
     /, ~\                / /
    X     `-.....-------./ /
     ~-. ~  ~              |
        \     Optim   /    |
         \  /_     ___\   /
         | /\ ~~~~~   \ |
         | | \        || |
         | |\ \       || )
        (_/ (_/      ((_/
  '''

  def __init__(self, sigWeights, sigMass, sigDiscrims, bkgWeights, bkgMass, bkgDiscrims, nCats, ranges, names):
    '''Initialise with the signal and background weights (as np arrays), then three lists: the discriminator arrays, the ranges (in the form [low, high]) and the names'''
    self.sigWeights    = sigWeights
    self.sigMass       = sigMass
    self.bkgWeights    = bkgWeights
    self.bkgMass       = bkgMass
    self.nonSigWeights = None
    self.nonSigMass    = None
    self.nCats         = int(nCats)
    self.bests         = Bests(self.nCats)
    self.sortOthers    = False
    self.addNonSig     = False
    self.transform     = False
    self.constantBkg   = False
    assert len(bkgDiscrims) == len(sigDiscrims)
    assert len(ranges)      == len(sigDiscrims)
    assert len(names)       == len(sigDiscrims)
    self.names          = names
    self.sigDiscrims    = od()
    self.bkgDiscrims    = od()
    self.nonSigDiscrims = None
    self.lows           = od()
    self.highs          = od()
    self.boundaries     = od()
    for iName,name in enumerate(self.names):
      self.sigDiscrims[ name ] = sigDiscrims[iName]
      self.bkgDiscrims[ name ] = bkgDiscrims[iName]
      assert len(ranges[iName]) == 2
      self.lows[ name ]       = ranges[iName][0]
      self.highs[ name ]      = ranges[iName][1]
      self.boundaries[ name ] = [-999. for i in range(self.nCats)]

  def setNonSig(self, nonSigWeights, nonSigMass, nonSigDiscrims):
    self.addNonSig      = True
    self.nonSigWeights  = nonSigWeights
    self.nonSigMass     = nonSigMass
    self.nonSigDiscrims = od()
    for iName,name in enumerate(self.names):
      self.nonSigDiscrims[ name ] = nonSigDiscrims[iName]

  def setTransform( self, val ):
    self.transform = val

  def doTransform( self, arr ):
    arr = 1. / ( 1. + np.exp( 0.5*np.log( 2./(arr+1.) - 1. ) ) )
    return arr

  def setConstantBkg( self, val ):
    self.constantBkg = val

  def optimise(self, lumi, nIters):
    '''Run the optimisation for a given number of iterations'''
    for iIter in range(nIters):
      cuts = od()
      for iName,name in enumerate(self.names):
        tempCuts = np.random.uniform(self.lows[name], self.highs[name], self.nCats)
        if iName==0 or self.sortOthers:
          tempCuts.sort()
        cuts[name] = tempCuts
        #if self.transform:
        #  tempCuts = self.doTransform(tempCuts)
      sigs = []
      bkgs = []
      nons = []
      for iCat in range(self.nCats):
        lastCat = (iCat+1 == self.nCats)
        sigWeights = self.sigWeights
        bkgWeights = self.bkgWeights
        if self.addNonSig: nonSigWeights = self.nonSigWeights
        for iName,name in enumerate(self.names):
          sigWeights = sigWeights * (self.sigDiscrims[name]>cuts[name][iCat])
          bkgWeights = bkgWeights * (self.bkgDiscrims[name]>cuts[name][iCat])
          if self.addNonSig: nonSigWeights = nonSigWeights * (self.nonSigDiscrims[name]>cuts[name][iCat])
          if not lastCat:
            if iName==0 or self.sortOthers:
              sigWeights = sigWeights * (self.sigDiscrims[name]<cuts[name][iCat+1])
              bkgWeights = bkgWeights * (self.bkgDiscrims[name]<cuts[name][iCat+1])
              if self.addNonSig: nonSigWeights = nonSigWeights * (self.nonSigDiscrims[name]<cuts[name][iCat+1])
        sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
        fill_hist(sigHist, self.sigMass, weights=sigWeights)
        sigCount = 0.68 * lumi * sigHist.Integral() 
        sigWidth = self.getRealSigma(sigHist)
        bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
        fill_hist(bkgHist, self.bkgMass, weights=bkgWeights)
        bkgCount = self.computeBkg(bkgHist, sigWidth)
        if self.addNonSig:
          nonSigHist = r.TH1F('nonSigHistTemp','nonSigHistTemp',160,100,180)
          fill_hist(nonSigHist, self.nonSigMass, weights=nonSigWeights)
          nonSigCount = 0.68 * lumi * nonSigHist.Integral() 
        else:
          nonSigCount = 0.
        sigs.append(sigCount)
        bkgs.append(bkgCount)
        nons.append(nonSigCount)
      if self.bests.update(sigs, bkgs, nons):
        for name in self.names:
          self.boundaries[name] = cuts[name]

  def crossCheck(self, lumi, plotDir):
    '''Run a check to ensure the random search found a good mimimum'''
    for iName,name in enumerate(self.names):
      for iCat in range(self.nCats):
        best = self.boundaries[name][iCat]
        rnge = 0.2 * self.highs[name] - self.lows[name]
        graph = r.TGraph()
        for iVal,val in enumerate(np.arange(best-rnge/2., best+rnge/2., rnge/10.)):
          sigs = []
          bkgs = []
          nons = []
          cuts = {} 
          cuts[name] = self.boundaries[name]
          cuts[name][iCat] = val
          bests = Bests(self.nCats)
          for jCat in range(self.nCats):
            lastCat = (jCat+1 == self.nCats)
            sigWeights = self.sigWeights
            bkgWeights = self.bkgWeights
            if self.addNonSig: nonSigWeights = self.nonSigWeights
            for jName,jname in enumerate(self.names):
              sigWeights = sigWeights * (self.sigDiscrims[jname]>cuts[jname][jCat])
              bkgWeights = bkgWeights * (self.bkgDiscrims[jname]>cuts[jname][jCat])
              if self.addNonSig: nonSigWeights = nonSigWeights * (self.nonSigDiscrims[jname]>cuts[jname][jCat])
              if not lastCat:
                if jName==0 or self.sortOthers:
                  sigWeights = sigWeights * (self.sigDiscrims[jname]<cuts[jname][jCat+1])
                  bkgWeights = bkgWeights * (self.bkgDiscrims[jname]<cuts[jname][jCat+1])
                  if self.addNonSig: nonSigWeights = nonSigWeights * (self.nonSigDiscrims[jname]<cuts[jname][jCat+1])
            sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
            fill_hist(sigHist, self.sigMass, weights=sigWeights)
            sigCount = 0.68 * lumi * sigHist.Integral() 
            sigWidth = self.getRealSigma(sigHist)
            bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
            fill_hist(bkgHist, self.bkgMass, weights=bkgWeights)
            bkgCount = self.computeBkg(bkgHist, sigWidth)
            if self.addNonSig:
              nonSigHist = r.TH1F('nonSigHistTemp','nonSigHistTemp',160,100,180)
              fill_hist(nonSigHist, self.nonSigMass, weights=nonSigWeights)
              nonSigCount = 0.68 * lumi * nonSigHist.Integral() 
            else:
              nonSigCount = 0.
            sigs.append(sigCount)
            bkgs.append(bkgCount)
            nons.append(nonSigCount)
          bests.update(sigs, bkgs, nons)
          graph.SetPoint(iVal, val-best, bests.getTotSignif())
        canv = useSty.setCanvas()
        graphName = 'CrossCheck_%s_Cat%g'%(name, iCat)
        graph.SetTitle(graphName.replace('_',' '))
        graph.GetXaxis().SetTitle('Cut value - chosen value')
        graph.GetYaxis().SetTitle('Significance (#sigma)')
        graph.Draw()
        useSty.drawCMS(text='Internal')
        useSty.drawEnPu(lumi=lumi)
        canv.Print('%s/%s.pdf'%(plotDir,graphName))
        canv.Print('%s/%s.png'%(plotDir,graphName))

  def setSortOthers(self, val):
    self.sortOthers = val

  def getBests(self):
    return self.bests

  def getPrintableResult(self):
    printStr = ''
    for iCat in reversed(range(self.nCats)):
      catNum = self.nCats - (iCat+1)
      printStr += 'Category %g optimal cuts are:  '%catNum
      for name in self.names:
        printStr += '%s %1.3f,  '%(name, self.boundaries[name][iCat])
      printStr = printStr[:-3]
      printStr += '\n'
      printStr += 'With  S %1.3f,  B %1.3f + %1.3f,  signif = %1.3f \n'%(self.bests.sigs[iCat], self.bests.bkgs[iCat], self.bests.nons[iCat], self.bests.signifs[iCat])
    printStr += 'Corresponding to a total significance of  %1.3f \n\n'%self.bests.totSignif
    return printStr

  def getRealSigma( self, hist ):
    sigma = 2.
    if hist.GetEntries() > 0 and hist.Integral()>0.000001:
      hist.Fit('gaus')
      fit = hist.GetFunction('gaus')
      sigma = fit.GetParameter(2)
    return sigma
  
  def computeBkg( self, hist, effSigma ):
    bkgVal = 9999.
    if hist.GetEntries() > 10 and hist.Integral()>0.000001:
      if self.constBkg:
        totalBkg = hist.Integral( hist.FindBin(100.1), hist.FindBin(119.9) ) + hist.Integral( hist.FindBin(130.1), hist.FindBin(179.9) )
        bkgVal = (totalBkg/70.) * 2. * effSigma
      else:
        hist.Fit('expo')
        fit = hist.GetFunction('expo')
        bkgVal = fit.Integral(125. - effSigma, 125. + effSigma)
    return bkgVal

  def getAMS(self, s, b, breg=3.):
    b = b + breg
    val = 0.
    if b > 0.:
      val = (s + b)*np.log(1. + (s/b))
      val = 2*(val - s)
      val = np.sqrt(val)
    return val

