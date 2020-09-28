# Author: Izaak Neutelings (May 2020)
# Description: Simple module to pre-select mutau events
from ROOT import TFile, TTree, TH1D
from ROOT import Math
from ROOT import TLorentzVector, TVector3
from math import sqrt, exp, cos, pi
import numpy as np
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from TauFW.PicoProducer.analysis.utils import deltaPhi
from TauFW.PicoProducer.corrections.PileupTool import *
from TauFW.PicoProducer.corrections.RecoilCorrectionTool import *
from TauFW.PicoProducer.corrections.MuonSFs import *


# Inspired by 'Object' class from NanoAODTools.
# Convenient to do so to be able to add MET as 4-momentum to other physics objects using p4()
class Met(Object):
  def __init__(self,event,prefix,index=None):
    self.eta = 0.0
    self.mass = 0.0
    Object.__init__(self,event,prefix,index)


class ModuleMuTau(Module):
  
  def __init__(self,fname,**kwargs):
    self.outfile = TFile(fname,'RECREATE')
    self.default_float = -999.0
    self.default_int = -999
    self.dtype      = kwargs.get('dtype', 'data')
    self.ismc       = self.dtype=='mc'
    self.isdata     = self.dtype=='data'
    self.filename   = fname
    self.year       = kwargs.get('year',    2018           ) # integer, e.g. 2017, 2018
    self.era        = kwargs.get('era',     '2017'         ) # string, e.g. '2017', 'UL2017'
    self.dozpt      = kwargs.get('zpt',     'DY' in fname  ) # Z pT reweighting
    self.verbosity  = kwargs.get('verb',    0              ) # verbosity

  def beginJob(self):
    """Prepare output analysis tree and cutflow histogram."""
    
    # CUTFLOW HISTOGRAM
    self.cutflow           = TH1D('cutflow','cutflow',25,0,25)
    self.cut_none          = 0
    self.cut_trig          = 1
    self.cut_muon          = 2
    self.cut_muon_veto     = 3
    self.cut_tau           = 4
    self.cut_electron_veto = 5
    self.cut_pair          = 6
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_none,           "no cut"        )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_trig,           "trigger"       )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_muon,           "muon"          )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_muon_veto,      "muon     veto" )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_tau,            "tau"           )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_electron_veto,  "electron veto" )
    self.cutflow.GetXaxis().SetBinLabel(1+self.cut_pair,           "pair"          )
    
    # TREE
    self.tree        = TTree('tree','tree')
    self.pt_1        = np.zeros(1,dtype='f')
    self.eta_1       = np.zeros(1,dtype='f')
    self.q_1         = np.zeros(1,dtype='i')
    self.id_1        = np.zeros(1,dtype='?')
    self.iso_1       = np.zeros(1,dtype='f')
    self.genmatch_1  = np.zeros(1,dtype='f')
    self.decayMode_1 = np.zeros(1,dtype='i')
    self.pt_2        = np.zeros(1,dtype='f')
    self.eta_2       = np.zeros(1,dtype='f')
    self.q_2         = np.zeros(1,dtype='i')
    self.id_2        = np.zeros(1,dtype='i')
    self.anti_e_2    = np.zeros(1,dtype='i')
    self.anti_mu_2   = np.zeros(1,dtype='i')
    self.iso_2       = np.zeros(1,dtype='f')
    self.genmatch_2  = np.zeros(1,dtype='f')
    self.decayMode_2 = np.zeros(1,dtype='i')
    self.m_vis       = np.zeros(1,dtype='f')
    self.genWeight   = np.zeros(1,dtype='f')
    self.zptweight   = np.zeros(1,dtype='f')
    self.puweight    = np.zeros(1,dtype='f')
    self.mu_SF_weight= np.zeros(1,dtype='f')

    self.njets       = np.zeros(1,dtype='i')
    self.nbjets      = np.zeros(1,dtype='i')
    self.jet_pt_1    = np.zeros(1,dtype='f')
    self.jet_eta_1   = np.zeros(1,dtype='f')
    self.jet_pt_2    = np.zeros(1,dtype='f')
    self.jet_eta_2   = np.zeros(1,dtype='f')
    self.bjet_pt_1   = np.zeros(1,dtype='f')
    self.bjet_eta_1  = np.zeros(1,dtype='f')
    self.bjet_pt_2   = np.zeros(1,dtype='f')
    self.bjet_eta_2  = np.zeros(1,dtype='f')
    self.pt_vis      = np.zeros(1,dtype='f')
    self.pt_Z_puppimet   = np.zeros(1,dtype='f')
    self.pt_Z_PFmet      = np.zeros(1,dtype='f')
    self.mt_1_puppimet   = np.zeros(1,dtype='f')
    self.mt_1_PFmet      = np.zeros(1,dtype='f')
    self.mt_2_puppimet   = np.zeros(1,dtype='f')
    self.mt_2_PFmet      = np.zeros(1,dtype='f')
    self.dzeta_puppimet  = np.zeros(1,dtype='f')
    self.dzeta_PFmet     = np.zeros(1,dtype='f')
    self.dR_ll           = np.zeros(1,dtype='f')
    self.rho             = np.zeros(1,dtype='f')
    self.npv             = np.zeros(1,dtype='i')
    self.npv_good        = np.zeros(1,dtype='i')
    self.npu             = np.zeros(1,dtype='i')
    self.npu_true        = np.zeros(1,dtype='f')

    self.tree.Branch('pt_1',         self.pt_1,        'pt_1/F'       )
    self.tree.Branch('eta_1',        self.eta_1,       'eta_1/F'      )
    self.tree.Branch('q_1',          self.q_1,         'q_1/I'        )
    self.tree.Branch('id_1',         self.id_1,        'id_1/O'       )
    self.tree.Branch('iso_1',        self.iso_1,       'iso_1/F'      )
    self.tree.Branch('genmatch_1',   self.genmatch_1,  'genmatch_1/F' )
    self.tree.Branch('decayMode_1',  self.decayMode_1, 'decayMode_1/I')
    self.tree.Branch('pt_2',         self.pt_2,  'pt_2/F'             )
    self.tree.Branch('eta_2',        self.eta_2, 'eta_2/F'            )
    self.tree.Branch('q_2',          self.q_2,   'q_2/I'              )
    self.tree.Branch('id_2',         self.id_2,  'id_2/I'             )
    self.tree.Branch('anti_e_2',     self.anti_e_2,   'anti_e_2/I'    )
    self.tree.Branch('anti_mu_2',    self.anti_mu_2,  'anti_mu_2/I'   )
    self.tree.Branch('iso_2',        self.iso_2, 'iso_2/F'            )
    self.tree.Branch('genmatch_2',   self.genmatch_2,  'genmatch_2/F' )
    self.tree.Branch('decayMode_2',  self.decayMode_2, 'decayMode_2/I')
    self.tree.Branch('m_vis',        self.m_vis, 'm_vis/F'            )
    self.tree.Branch('genWeight',    self.genWeight,   'genWeight/F'  )
    self.tree.Branch('zptweight',    self.zptweight,   'zptweight/F'  )
    self.tree.Branch('puweight',     self.puweight,   'puweight/F'  )
    self.tree.Branch('mu_SF_weight', self.mu_SF_weight,   'mu_SF_weight/F'  )

    self.tree.Branch('njets',          self.njets,     'njets/I'                   )
    self.tree.Branch('nbjets',         self.nbjets,     'nbjets/I'                 )
    self.tree.Branch('jet_pt_1',       self.jet_pt_1,     'jet_pt_1/F'             )
    self.tree.Branch('jet_eta_1',      self.jet_eta_1,     'jet_eta_1/F'           )
    self.tree.Branch('jet_pt_2',       self.jet_pt_2,     'jet_pt_2/F'             )
    self.tree.Branch('jet_eta_2',      self.jet_eta_2,     'jet_eta_2/F'           )
    self.tree.Branch('bjet_pt_1',      self.bjet_pt_1,     'bjet_pt_1/F'           )
    self.tree.Branch('bjet_eta_1',     self.bjet_eta_1,     'bjet_eta_1/F'         )
    self.tree.Branch('bjet_pt_2',      self.bjet_pt_2,     'bjet_pt_2/F'           )
    self.tree.Branch('bjet_eta_2',     self.bjet_eta_2,     'bjet_eta_2/F'         )
    self.tree.Branch('pt_vis',         self.pt_vis,     'pt_vis/F'                 )
    self.tree.Branch('pt_Z_puppimet',  self.pt_Z_puppimet,     'pt_Z_puppimet/F'   )
    self.tree.Branch('pt_Z_PFmet',     self.pt_Z_PFmet,     'pt_Z_PFmet/F'         )
    self.tree.Branch('mt_1_puppimet',  self.mt_1_puppimet,     'mt_1_puppimet/F'   )
    self.tree.Branch('mt_1_PFmet',     self.mt_1_PFmet,     'mt_1_PFmet/F'         )
    self.tree.Branch('mt_2_puppimet',  self.mt_2_puppimet,     'mt_2_puppimet/F'   )
    self.tree.Branch('mt_2_PFmet',     self.mt_2_PFmet,     'mt_2_PFmet/F'         )
    self.tree.Branch('dzeta_puppimet', self.dzeta_puppimet,     'dzeta_puppimet/F' )
    self.tree.Branch('dzeta_PFmet',    self.dzeta_PFmet,     'dzeta_PFmet/F'       )
    self.tree.Branch('dR_ll',          self.dR_ll,     'dR_ll/F'                   )
    self.tree.Branch('rho',            self.rho,     'rho/F'                       )
    self.tree.Branch('npv',            self.npv,     'npv/I'                       )
    self.tree.Branch('npv_good',       self.npv_good,     'npv_good/I'             )
    self.tree.Branch('npu',            self.npu,     'npu/I'                       )
    self.tree.Branch('npu_true',       self.npu_true,     'npu_true/F'             )
  
    # get corrections
    if self.ismc:
      self.muSFs          = MuonSFs(era=self.era,verb=self.verbosity)
      self.puTool         = PileupWeightTool(era=self.era,sample=self.filename,verb=self.verbosity)
      if self.dozpt:
        self.zptTool      = ZptCorrectionTool(year=self.year)
  
  def endJob(self):
    """Wrap up after running on all events and files"""
    self.outfile.Write()
    self.outfile.Close()
  
  def analyze(self, event):
    """Process event, return True (pass, go to next module) or False (fail, go to next event)."""
    
    # NO CUT
    self.cutflow.Fill(self.cut_none)
    
    # TRIGGER
    if not event.HLT_IsoMu27: return False
    self.cutflow.Fill(self.cut_trig)
    
    # SELECT MUON
    muons = [ ]
    # TODO section 4: extend with a veto of additional muons. Veto muons should have the same quality selection as signal muons (or even looser),
    # but with a lower pt cut, e.g. muon.pt > 15.0
    veto_muons = [ ]

    for muon in Collection(event,'Muon'):
      good_muon = muon.mediumId and muon.pfRelIso04_all < 0.5 and abs(muon.eta) < 2.5
      signal_muon = good_muon and muon.pt > 28.0
      veto_muon   = good_muon and muon.pt > 15.0 # TODO section 4: introduce a veto muon selection here
      if signal_muon:
        muons.append(muon)
      if veto_muon: # CAUTION: that's NOT an elif here and intended in that way!
        veto_muons.append(muon)
     
    if len(muons) == 0: return False
    self.cutflow.Fill(self.cut_muon)
    # TODO section 4: What should be the requirement to veto events with additional muons?
    if len(veto_muons) > len(muons): return False
    self.cutflow.Fill(self.cut_muon_veto)
    
    # SELECT TAU
    # TODO section 6: Which decay modes of a tau should be considered for an analysis? Extend tau selection accordingly
    taus = [ ]
    for tau in Collection(event,'Tau'):
      good_tau = tau.pt > 18.0 and tau.idDeepTau2017v2p1VSe >= 1 and tau.idDeepTau2017v2p1VSmu >= 1 and tau.idDeepTau2017v2p1VSjet >= 1
      if good_tau:
        taus.append(tau)
    if len(taus)<1: return False
    self.cutflow.Fill(self.cut_tau)

    # SELECT ELECTRONS FOR VETO
    # TODO section 4: extend the selection of veto electrons: pt > 15.0,
    # with loose WP of the mva based ID (Fall17 training without isolation),
    # and a custom isolation cut on PF based isolation using all PF candidates.
    electrons = []
    for electron in Collection(event,'Electron'):
      veto_electron = electron.mvaFall17V2Iso_WPL and electron.pt > 15.0 and electron.pfRelIso03_all < 0.5 # TODO section 4: introduce a veto electron selection here
      if veto_electron:
        electrons.append(electron)
    if len(electrons) > 0: return False
    self.cutflow.Fill(self.cut_electron_veto)
    
    # PAIR
    # TODO section 4 (optional): the mutau pair is constructed from a muon with highest pt and a tau with highest pt.
    # However, there is also the possibility to select the mutau pair according to the isolation.
    # If you like, you could try to implement mutau pair building algorithm, following the instructions on
    # https://twiki.cern.ch/twiki/bin/view/CMS/HiggsToTauTauWorking2017#Pair_Selection_Algorithm, but using the latest isolation quantities/discriminators
    muon = max(muons,key=lambda p: p.pt)
    tau  = max(taus,key=lambda p: p.pt)
    if muon.DeltaR(tau)<0.4: return False
    self.cutflow.Fill(self.cut_pair)

    # SELECT Jets
    # TODO section 4: Jets are not used directly in our analysis, but it can be good to have a look at least the number of jets (and b-tagged jets) of your selection.
    # Therefore, collect at first jets with pt > 20, |eta| < 4.7, passing loose WP of Pileup ID, and tight WP for jetID.
    # The collected jets are furthermore not allowed to overlap with the signal muon and signal tau in deltaR, so selected them to have deltaR >= 0.5 w.r.t. the signal muon and signal tau.
    jets = [ ]
    bjets = [ ]
    for jet in Collection(event,'Jet'):
      good_jet = jet.pt > 20.0 and abs(jet.eta) < 4.7 and jet.puId >= 4 and jet.jetId >= 2 and jet.DeltaR(tau) >= 0.5 and jet.DeltaR(muon) >= 0.5
    # Then, select for this collection "usual" jets, which have pt > 30 in addition, count their number, and store pt & eta of the leading and subleading jet.
      if good_jet and jet.pt > 30.0:
        jets.append(jet)
    # For b-tagged jets, require additionally DeepFlavour b+bb+lepb tag with medium WP and |eta| < 2.5, count their number, and store pt & eta of the leading and subleading b-tagged jet.
      if good_jet and jet.btagDeepFlavB > 0.2770 and abs(jet.eta) < 2.5:  # btag WP from: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
        bjets.append(jet)
    jets.sort(key=lambda p: p.pt,reverse=True)
    bjets.sort(key=lambda p: p.pt,reverse=True)
    self.njets[0]       = len(jets)
    self.nbjets[0]      = len(bjets)
    if len(jets) > 0:
      self.jet_pt_1[0]  = jets[0].pt
      self.jet_eta_1[0] = jets[0].eta
    else:
      self.jet_pt_1[0]  = -1.
      self.jet_eta_1[0] = -10.
    if len(jets) > 1:
      self.jet_pt_2[0]  = jets[1].pt
      self.jet_eta_2[0] = jets[1].eta
    else:
      self.jet_pt_2[0]  = -1.
      self.jet_eta_2[0] = -10.

    if len(bjets) > 0:
      self.bjet_pt_1[0]  = bjets[0].pt
      self.bjet_eta_1[0] = bjets[0].eta
    else:
      self.bjet_pt_1[0]  = -1.
      self.bjet_eta_1[0] = -10.
    if len(bjets) > 1:
      self.bjet_pt_2[0]  = bjets[1].pt
      self.bjet_eta_2[0] = bjets[1].eta
    else:
      self.bjet_pt_2[0]  = -1.
      self.bjet_eta_2[0] = -10.

    # CHOOSE MET definition
    # TODO section 4: compare the PuppiMET and (PF-based) MET in terms of mean, resolution and data/expectation agreement of their own distributions and of related quantities
    # and choose one of them for the refinement of Z to tautau selection.
    puppimet = Met(event, 'PuppiMET')
    met = Met(event, 'MET')
    

    # SAVE VARIABLES
    # TODO section 4: extend the variable list with more quantities (also high level ones). Compute at least:
    # - visible pt of the Z boson candidate
    # - best-estimate for pt of Z boson candidate (now including contribution form neutrinos)
    # - transverse mass of the system composed from the muon and MET vectors. Definition can be found in doi:10.1140/epjc/s10052-018-6146-9.
    #   Caution: use ROOT DeltaPhi for difference in phi and check that deltaPhi is between -pi and pi.Have a look at transverse mass with both versions of MET
    # - Dzeta. Definition can be found in doi:10.1140/epjc/s10052-018-6146-9. Have a look at the variable with both versions of MET
    # - Separation in DeltaR between muon and tau candidate
    # - global event quantities like the proper definition of pileup density rho, number of reconstructed vertices,
    # - in case of MC: number of true (!!!) pileup interactions
    self.pt_1[0]        = muon.pt
    self.eta_1[0]       = muon.eta
    self.q_1[0]         = muon.charge
    self.id_1[0]        = muon.mediumId
    self.iso_1[0]       = muon.pfRelIso04_all # keep in mind: the SMALLER the value, the more the muon is isolated
    self.decayMode_1[0] = self.default_int # not needed for a muon
    self.pt_2[0]        = tau.pt
    self.eta_2[0]       = tau.eta
    self.q_2[0]         = tau.charge
    self.id_2[0]        = tau.idDeepTau2017v2p1VSjet
    self.anti_e_2[0]    = tau.idDeepTau2017v2p1VSe
    self.anti_mu_2[0]   = tau.idDeepTau2017v2p1VSmu
    self.iso_2[0]       = tau.rawDeepTau2017v2p1VSjet # keep in mind: the HIGHER the value of the discriminator, the more the tau is isolated
    self.decayMode_2[0] = tau.decayMode
    self.m_vis[0]       = (muon.p4()+tau.p4()).M()
    self.pt_vis[0]      = (muon.p4()+tau.p4()).Pt()
    self.pt_Z_puppimet[0]      = (muon.p4()+tau.p4()+puppimet.p4()).Pt()
    self.pt_Z_PFmet[0]         = (muon.p4()+tau.p4()+met.p4()).Pt()
    self.mt_1_puppimet[0]      = sqrt( 2*muon.pt*puppimet.pt*(1-cos(deltaPhi(muon.phi,puppimet.phi))) )
    self.mt_1_PFmet[0]         = sqrt( 2*muon.pt*met.pt*(1-cos(deltaPhi(muon.phi,met.phi))) )
    self.mt_2_puppimet[0]      = sqrt( 2*tau.pt*puppimet.pt*(1-cos(deltaPhi(tau.phi,puppimet.phi))) )
    self.mt_2_PFmet[0]         = sqrt( 2*tau.pt*met.pt*(1-cos(deltaPhi(tau.phi,met.phi))) )

    # calculate dZeta
    leg1                       = TVector3(muon.p4().Px(),muon.p4().Py(),0.)
    leg2                       = TVector3(tau.p4().Px(),tau.p4().Py(),0.)
    zetaAxis                   = TVector3(leg1.Unit()+leg2.Unit()).Unit()
    pzetavis                   = leg1*zetaAxis + leg2*zetaAxis
    pzetamiss_puppi            = puppimet.p4().Vect()*zetaAxis
    pzetamiss_PF               = met.p4().Vect()*zetaAxis
    self.dzeta_puppimet[0]     = pzetamiss_puppi - 0.85*pzetavis
    self.dzeta_PFmet[0]        = pzetamiss_PF - 0.85*pzetavis
    self.dR_ll[0]              = muon.DeltaR(tau)
    self.rho[0]                = event.fixedGridRhoFastjetAll
    self.npv[0]                = event.PV_npvs
    self.npv_good[0]           = event.PV_npvsGood

    if self.ismc:
      self.genmatch_1[0]  = muon.genPartFlav # in case of muons: 1 == prompt muon, 15 == muon from tau decay, also other values available for muons from jets
      self.genmatch_2[0]  = tau.genPartFlav # in case of taus: 0 == unmatched (corresponds then to jet),
                                            #                  1 == prompt electron, 2 == prompt muon, 3 == electron from tau decay,
                                            #                  4 == muon from tau decay, 5 == hadronic tau decay
      self.genWeight[0] = event.genWeight
      self.npu[0]                = event.Pileup_nPU
      self.npu_true[0]           = event.Pileup_nTrueInt
    # get corrections
      self.mu_SF_weight[0]       = self.muSFs.getIdIsoSF(muon.pt,muon.eta)
      self.puweight[0]           = self.puTool.getWeight(event.Pileup_nTrueInt)
      if self.dozpt:
        truthZ = []
        for genPart in Collection(event,'GenPart'):
          truthZboson = (genPart.pdgId == 23) and (genPart.status == 62)
          if truthZboson:
            truthZ.append(genPart)
        if len(truthZ)==1: 
          self.zptweight[0]        = self.zptTool.getZptWeight(truthZ[0].p4().Pt(),truthZ[0].p4().M())
    self.tree.Fill()
    
    return True
