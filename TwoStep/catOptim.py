from collections import OrderedDict as od
import numpy as np
import ROOT as r
r.gROOT.SetBatch(True)
from root_numpy import fill_hist


class Bests:
  '''Class to store and update best values during a category optimisation'''

  def __init__(self, nCats):
    self.nCats = nCats

    self.totSignif = -999.
    self.sigs      = [-999. for i in range(self.nCats)]
    self.bkgs      = [-999. for i in range(self.nCats)]
    self.signifs   = [-999. for i in range(self.nCats)]

  def update(self, sigs, bkgs):
    signifs = []
    totSignifSq = 0.
    for i in range(self.nCats):
      sig = sigs[i] 
      bkg = bkgs[i] 
      signif = self.getAMS(sig, bkg)
      signifs.append(signif)
      totSignifSq += signif*signif
    totSignif = np.sqrt( totSignifSq )
    if totSignif > self.totSignif:
      self.totSignif = totSignif
      for i in range(self.nCats):
        self.sigs[i]     = sigs[i]
        self.bkgs[i]     = bkgs[i]
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
    self.sigWeights  = sigWeights
    self.sigMass     = sigMass
    self.bkgWeights  = bkgWeights
    self.bkgMass     = bkgMass
    self.nCats       = int(nCats)
    self.bests       = Bests(self.nCats)
    self.sortOthers  = False
    assert len(bkgDiscrims) == len(sigDiscrims)
    assert len(ranges)      == len(sigDiscrims)
    assert len(names)       == len(sigDiscrims)
    self.names       = names
    self.sigDiscrims = od()
    self.bkgDiscrims = od()
    self.lows        = od()
    self.highs       = od()
    self.boundaries  = od()
    for iName,name in enumerate(self.names):
      self.sigDiscrims[ name ] = sigDiscrims[iName]
      self.bkgDiscrims[ name ] = bkgDiscrims[iName]
      assert len(ranges[iName]) == 2
      self.lows[ name ]       = ranges[iName][0]
      self.highs[ name ]      = ranges[iName][1]
      self.boundaries[ name ] = [-999. for i in range(self.nCats)]

  def optimise(self, lumi, nIters):
    '''Run the optimisation for a given number of iterations'''
    for iIter in range(nIters):
      cuts = od()
      for iName,name in enumerate(self.names):
        tempCuts = np.random.uniform(self.lows[name], self.highs[name], self.nCats)
        if iName==0 or self.sortOthers:
          tempCuts.sort()
        cuts[name] = tempCuts
      sigs = []
      bkgs = []
      for iCat in range(self.nCats):
        lastCat = (iCat+1 == self.nCats)
        sigWeights = self.sigWeights
        bkgWeights = self.bkgWeights
        for iName,name in enumerate(self.names):
          sigWeights = sigWeights * (self.sigDiscrims[name]>cuts[name][iCat])
          bkgWeights = bkgWeights * (self.bkgDiscrims[name]>cuts[name][iCat])
          if not lastCat:
            if iName==0 or self.sortOthers:
              sigWeights = sigWeights * (self.sigDiscrims[name]<cuts[name][iCat+1])
              bkgWeights = bkgWeights * (self.bkgDiscrims[name]<cuts[name][iCat+1])
        sigHist = r.TH1F('sigHistTemp','sigHistTemp',160,100,180)
        fill_hist(sigHist, self.sigMass, weights=sigWeights)
        sigCount = 0.68 * lumi * sigHist.Integral() 
        sigWidth = self.getRealSigma(sigHist)
        bkgHist = r.TH1F('bkgHistTemp','bkgHistTemp',160,100,180)
        fill_hist(bkgHist, self.bkgMass, weights=bkgWeights)
        bkgCount = self.computeBkg(bkgHist, sigWidth)
        sigs.append(sigCount)
        bkgs.append(bkgCount)
      if self.bests.update(sigs, bkgs):
        for name in self.names:
          self.boundaries[name] = cuts[name]

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
      printStr += 'With  S %1.3f,  B %1.3f,  signif = %1.3f \n'%(self.bests.sigs[iCat], self.bests.bkgs[iCat], self.bests.signifs[iCat])
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
    if hist.GetEntries() > 0 and hist.Integral()>0.000001:
      hist.Fit('expo')
      fit = hist.GetFunction('expo')
      bkgVal = fit.Integral(125. - effSigma, 125. + effSigma)
    return bkgVal
