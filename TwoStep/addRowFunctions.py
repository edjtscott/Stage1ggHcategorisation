def addPt(row):
    return row['CMS_hgg_mass']*row['diphoptom']

def truthDipho(row):
    if not row['HTXSstage1p2bin']==0: return 1
    else: return 0

def truthVhHad(row):
    if row['HTXSstage1p2bin']==204: return 1
    else: return 0

def truthVhHadForWeights(row):
    if row['HTXSstage1p2bin']==204: return 2
    elif row['HTXSstage1p2bin']>99.5 and row['HTXSstage1p2bin']<116.5: return 1
    else: return 0

def vhHadWeight(row, vhRatio, gghRatio):
    weight = abs(row['weight'])
    if row['HTXSstage1p2bin']>199.5 and row['HTXSstage1p2bin']<203.5: 
        weight = 0.
    elif row['HTXSstage1p2bin']>204.5:
        weight = 0.
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    if row['truthVhHadForWeights']==2: 
      return vhRatio * weight
    elif row['truthVhHadForWeights']==1: 
      return gghRatio * weight
    else: return weight

def truthVBF(row):
    if row['HTXSstage1p2bin']>206.5 and row['HTXSstage1p2bin']<210.5: return 2
    elif row['HTXSstage1p2bin']>112.5 and row['HTXSstage1p2bin']<116.5: return 1
    else: return 0

def vbfWeight(row, vbfSumW, gghSumW, bkgSumW):
    weight = abs(row['weight'])
    if row['HTXSstage1p2bin']>99.5 and row['HTXSstage1p2bin']<112.5: 
        weight = 0.
    elif row['HTXSstage1p2bin']>199.5 and row['HTXSstage1p2bin']<206.5: 
        weight = 0.
    if row['proc'].count('qcd'): 
        weight *= 0.04 
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    if row['truthVBF']==2: 
      return (bkgSumW/vbfSumW) * weight
    elif row['truthVBF']==1: 
      return (bkgSumW/gghSumW) * weight
    else: return weight

def truthClass(row):
    if not row['HTXSstage1p2bin']==0: return int(row['HTXSstage1p2bin']-3)
    else: return 0

def truthJets(row):
    if row['HTXSstage1p2bin']==3: return 0
    elif row['HTXSstage1p2bin']>=4 and row['HTXSstage1p2bin']<=7: return 1
    elif row['HTXSstage1p2bin']>=8 and row['HTXSstage1p2bin']<=11: return 2
    else: return -1

def reco(row):
    if row['n_rec_jets']==0: return 0
    elif row['n_rec_jets']==1:
        if row['diphopt'] < 60: return 1
        elif row['diphopt'] < 120: return 2
        elif row['diphopt'] < 200: return 3
        else: return 4
    else:
        if row['diphopt'] < 60: return 5
        elif row['diphopt'] < 120: return 6
        elif row['diphopt'] < 200: return 7
        else: return 8

def diphoWeight(row, sigWeight=1.):
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    elif row['HTXSstage1p2bin'] > 0.01:
        weight *= sigWeight #arbitrary change in signal weight, to be optimised
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight

def combinedWeight(row):
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    weight = abs(weight)
    return weight

def normWeight(row, bkgWeight=100., zerojWeight=1.):
    weightFactors = [0.0002994, 0.0000757, 0.0000530, 0.0000099, 0.0000029, 0.0000154, 0.0000235, 0.0000165, 0.0000104] #update these at some point
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 / weightFactors[ int(row['truthClass']) ] #reduce because too large by default
    else: 
        weight *= 1. / weightFactors[ int(row['truthClass']) ] #otherwise just reweight by xs
    weight = abs(weight)
    #arbitrary weight changes to be optimised
    if row['proc'] != 'ggh':
        weight *= bkgWeight
    elif row['reco'] == 0: 
        weight *= zerojWeight
    return weight

def jetWeight(row):
    weightFactors = [0.606560, 0.270464, 0.122976]
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthJets']) ] #otherwise just reweight by xs
    weight = abs(weight)
    return weight

def altDiphoWeight(row, weightRatio):
    weight = row['weight']
    if row['proc'].count('qcd'):
        weight *= 0.04 #downweight bc too few events
    elif row['HTXSstage1p2bin'] > 0.01:
        weight *= weightRatio #arbitrary change in signal weight, to be optimised
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight

def cosThetaStar(row):
    from ROOT import TLorentzVector as lv
    from numpy import pi
    leadPho = lv()
    leadPho.SetPtEtaPhiM( row['dipho_lead_ptoM'] * row['dipho_mass'], row['dipho_leadEta'], row['dipho_leadPhi'], 0. )
    subleadPho = lv()
    subleadPho.SetPtEtaPhiM( row['dipho_sublead_ptoM'] * row['dipho_mass'], row['dipho_subleadEta'], row['dipho_subleadPhi'], 0. )

    diphoSystem = leadPho + subleadPho

    leadJet = lv()
    leadJetPhi = row['dijet_leadDeltaPhi'] + diphoSystem.Phi()
    if leadJetPhi > pi: leadJetPhi = leadJetPhi - 2.*pi
    elif leadJetPhi < -1.*pi: leadJetPhi = leadJetPhi + 2.*pi
    #print 'ED DEBUG leadJetPhi, dipho phi, deltaPhi = %.3f, %.3f, %.3f'%(leadJetPhi, diphoSystem.Phi(), row['dijet_leadDeltaPhi'])
    leadJet.SetPtEtaPhiM( row['dijet_LeadJPt'], row['dijet_leadEta'], leadJetPhi, 0. )
    subleadJet = lv()
    subleadJetPhi = row['dijet_subleadDeltaPhi'] + diphoSystem.Phi()
    if subleadJetPhi > pi: subleadJetPhi = subleadJetPhi - 2.*pi
    elif subleadJetPhi < -1.*pi: subleadJetPhi = subleadJetPhi + 2.*pi
    #print 'ED DEBUG subleadJetPhi, dipho phi, deltaPhi = %.3f, %.3f, %.3f'%(subleadJetPhi, diphoSystem.Phi(), row['dijet_subleadDeltaPhi'])
    subleadJet.SetPtEtaPhiM( row['dijet_SubJPt'], row['dijet_subleadEta'], subleadJetPhi, 0. )

    diphoDijetSystem = leadPho + subleadPho + leadJet + subleadJet

    diphoSystem.Boost( -diphoDijetSystem.BoostVector() )

    returnVal = -1. * diphoSystem.CosTheta()
    #print 'ED DEBUG resulting cos theta star value is %.3f'%returnVal
    
    return returnVal
