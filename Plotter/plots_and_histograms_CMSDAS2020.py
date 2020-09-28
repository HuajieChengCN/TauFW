#! /usr/bin/env python
# Author: Izaak Neutelings (August 2020)
# Description: Simple plotting script for pico analysis tuples
#   ./plot.py -c mutau -y 2018
import re
import ROOT as R
from TauFW.Plotter.plot.utils import LOG as PLOG
from TauFW.Plotter.sample.utils import LOG, STYLE, ensuredir, repkey, joincuts, setera, getyear, Sel, Var
from TauFW.Plotter.sample.utils import getsampleset as _getsampleset

def getsampleset(channel,era,**kwargs):
  verbosity = LOG.getverbosity(kwargs)
  year  = getyear(era) # get integer year
  tag   = kwargs.get('tag',   ""           )
  table = kwargs.get('table', True         ) # print sample set table
  setera(era) # set era for plot style and lumi normalization

  negative_fractions = {
    "DYJetsToLL_M-50" : 0.0004,
    "WJetsToLNu" : 0.0004,
    "WW_TuneCP5_13TeV-pythia8" : 0.0, # <-- no adaption of effective events needed
    "WZ_TuneCP5_13TeV-pythia8" : 0.0, # <-- no adaption of effective events needed
    "ZZ_TuneCP5_13TeV-pythia8" : 0.0, # <-- no adaption of effective events needed
    "ST_t-channel_top" : 0.033, 
    "ST_t-channel_antitop" : 0.031, 
    "ST_tW_top" : 0.002, 
    "ST_tW_antitop" : 0.002, 
    "TTTo2L2Nu" : 0.004,
    "TTToHadronic" : 0.004,
    "TTToSemiLeptonic" : 0.004,
  }
  
  # SM BACKGROUND MC SAMPLES
  expsamples = [ # table of MC samples to be converted to Sample objects
    # GROUP NAME                     TITLE                 XSEC [pb]      effective NEVENTS = simulated NEVENTS * ( 1  - 2 * negative fraction)

    # Cross-secitons: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV, Z/a* (50)
    #( 'DY', "DYJetsToLL_M-50",       "Drell-Yan 50",       6077.22,       {"nevts" : 100194597 * (1.0 - 2 * negative_fractions["DYJetsToLL_M-50"])} ),
    ( 'DY', "DYJetsToLL_M-50",       "Drell-Yan 50",       6077.22,       {"nevts" : 100194597 * (1.0 - 2 * negative_fractions["DYJetsToLL_M-50"]), 'extraweight': 'zptweight'} ),

    # Cross-sections: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV, Total W
    ( 'WJ', "WJetsToLNu",            "W + jets",           3*20508.9,     {"nevts" : 71072199 * (1.0 - 2 * negative_fractions["WJetsToLNu"])}),

    # Cross-sections: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV, W+ W-
    ( 'VV', "WW",                    "WW",                 118.7, {"nevts" : 7850000}),

    # Cross-sections: from generator (https://cms-gen-dev.cern.ch/xsdb with 'process_name=WZ_TuneCP5_13TeV-pythia8')
    ( 'VV', "WZ",                    "WZ",                 27.6, {"nevts" : 3885000}),

    # Cross-sections: from generator (https://cms-gen-dev.cern.ch/xsdb with 'process_name=ZZ_TuneCP5_13TeV-pythia8')
    ( 'VV', "ZZ",                    "ZZ",                 12.14, {"nevts" : 1979000}),

    # Cross-sections: # https://twiki.cern.ch/twiki/bin/viewauth/CMS/SingleTopSigma
    ( 'ST', "ST_t-channel_top",      "ST t-channel t",     136.02, {"nevts" : 154307600 * (1.0 - 2 * negative_fractions["ST_t-channel_top"])}),
    ( 'ST', "ST_t-channel_antitop",  "ST t-channel at",    80.95,  {"nevts" : 79090800 * (1.0 - 2 * negative_fractions["ST_t-channel_antitop"]) }),
    ( 'ST', "ST_tW_top",             "ST tW",              35.85,  {"nevts" : 9598000 * (1.0 - 2 * negative_fractions["ST_tW_top"])  }),
    ( 'ST', "ST_tW_antitop",         "ST atW",             35.85,  {"nevts" : 7623000 * (1.0 - 2 * negative_fractions["ST_tW_antitop"])  }),

    # Cross-sections: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO#Top_quark_pair_cross_sections_at, m_top = 172.5 GeV + PDG for W boson decays
    ( 'TT', "TTTo2L2Nu",             "ttbar 2l2#nu",       831.76*(3*0.1086)**2,         {"nevts" : 64310000 * (1.0 - 2 * negative_fractions["TTTo2L2Nu"]) }), 
    ( 'TT', "TTToHadronic",          "ttbar hadronic",     831.76*2*(3*0.1086)*(0.6741), {"nevts" : 199524000 * (1.0 - 2 * negative_fractions["TTToHadronic"])}),
    ( 'TT', "TTToSemiLeptonic",      "ttbar semileptonic", 831.76*(0.6741)**2,           {"nevts" : 199925998 * (1.0 - 2 * negative_fractions["TTToSemiLeptonic"])}),
  ]
  
  # OBSERVED DATA SAMPLES
  if 'mutau'  in channel: dataset = "SingleMuon_Run%d?"%year
  else:
    LOG.throw(IOError,"Did not recognize channel %r!"%(channel))
  datasample = ('Data',dataset) # Data for chosen channel
  
  # SAMPLE SET
  # TODO section 5: This weight needs to be extended with correction weights common to all simulated samples (MC)
  #weight = "genWeight/abs(genWeight)" # normalize weight, since sometimes the generator cross-section is contained in it.
  #weight = "puweight*genWeight/abs(genWeight)" # normalize weight, since sometimes the generator cross-section is contained in it.
  weight = "puweight*mu_isoSF_weight*mu_idSF_weight*genWeight/abs(genWeight)" # normalize weight, since sometimes the generator cross-section is contained in it.
  kwargs.setdefault('weight',weight)  # common weight for MC
  sampleset = _getsampleset(datasample,expsamples,channel=channel,era=era,**kwargs)
  
  # JOIN
  # Note: titles are set via STYLE.sample_titles
  sampleset.join('WW', 'WZ', 'ZZ', name='VV'  ) # Diboson
  sampleset.join('VV', 'WJ', name='EWK') # Electroweak

  sampleset.join('TT', name='TT' ) # ttbar
  sampleset.join('ST', name='ST' ) # single top
  sampleset.join('ST','TT', name='Top' ) #ttbar & single top 
  
  # SPLIT
  # TODO section 5: Check the generator matching for various samples in the flat n-tuples.
  # Is it justified to require only the tauh candidate to match to generator level hadronic tau to declare the full process with Z->tautau in mutau final state?
  # What is the major contribution from Drell-Yan to genmatch_2!=5? How does this look like for other processes?
  GMR = "genmatch_2==5"
  GMO = "genmatch_2!=5"
  sampleset.split('DY', [('ZTT',GMR),('ZL',GMO)])
  sampleset.split('Top',[('TopT',GMR),('TopJ',GMO)])
  sampleset.split('EWK',[('EWKT',GMR),('EWKJ',GMO)])
 
  if table:
    sampleset.printtable(merged=True,split=True)
  return sampleset


def plot(sampleset,channel,parallel=True,tag="",outdir="plots",histdir="",era=""):
  """Test plotting of SampleSet class for data/MC comparison."""
  LOG.header("plot")
  
  # SELECTIONS
  inclusive = "(q_1*q_2<0)"
  inclusive = inclusive.replace(" ","")
  #inclusive_cr_qcd = inclusive.replace("q_1*q_2<0","q_1*q_2>0") # inverting the opposite-sign requirement of the mutau pair into a same-sign requirment
  zregion   = "%s && mt_1_puppimet<60 && nbjets==0 && m_vis>40 && m_vis<100 && dzeta_puppimet>-50 "%(inclusive)
  selections = [
    Sel('inclusive',inclusive),
    Sel('zregion',zregion),
    #Sel('inclusive_cr_qcd',inclusive_cr_qcd),
  ]
  
  # VARIABLES
  # TODO section 5: extend with other variables, which are available in the flat n-tuples
  variables = [
     Var('m_vis',  40,  0, 200),
     Var('pt_vis',  "p_{T}^{vis} ", 40,  0, 200),
     Var('pt_Z_puppimet',  "p_{T}^{Z} with puppi", 40,  0, 200),
     #Var('pt_Z_PFmet',  "p_{T}^{Z} with PF", 40,  0, 200),
     #Var('m_2',   "m_tau_h",     28,  0.2, 1.6),
     Var('mt_1_puppimet',  "mt(mu,MET) with puppi", 40,  0, 200),
     #Var('mt_1_PFmet',  "mt(mu,MET) with PF", 40,  0, 200),
     Var('dzeta_puppimet',  "D_{zeta} with puppi",  56, -180, 100, pos='L;y=0.88',units='GeV'),
     #Var('dzeta_PFmet',  "D_{zeta} with PF",  56, -180, 100, pos='L;y=0.88',units='GeV'),
     Var('pt_1',  "Muon pt",    40,  0, 120, ),
     Var('pt_2',  "tau_h pt",   40,  0, 120, ),
     Var('eta_1', "Muon eta",   30, -3, 3, ymargin=1.6, ncols=2),
     Var('eta_2', "tau_h eta",  30, -3, 3, ymargin=1.6, ncols=2),
     Var('njets', 8,  0,  8),
     Var('nbjets', 8,  0,  8),
     Var('dR_ll',   "DR(mutau_h)",    30, 0, 6.0, ctitle={'etau':"DR(etau_h)",'tautau':"DR(tau_htau_h)",'emu':"DR(emu)"}),
     #Var('met_puppimet',    "p_{T}^{miss} with puppi",   50,  0, 150),
     #Var('met_PFmet',    "p_{T}^{miss} with PF",   50,  0, 150),
     Var('rho',    "#rho",    40,    0,  80 ),
     Var('npv',    40,    0,  80 ),
     Var('npv_good',    "npv_{good}",   40,    0,  80 ),
     Var('id_2',    "tau ID vs jet",     65,    0,    65 ),
     Var('anti_e_2',    "tau ID vs ele",     65,    0,    65 ),
     Var('anti_mu_2',    "tau ID vs muon",     20,    0,    20 ),
  ]
  
  # PLOT and HIST
  outdir   = ensuredir(repkey(outdir,CHANNEL=channel,ERA=era))
  histdir  = ensuredir(repkey(histdir,CHANNEL=channel,ERA=era))
  outhists = R.TFile.Open(histdir,'recreate')
  exts     = ['png']
  #exts     = ['png','pdf']
  for selection in selections:
    outhists.mkdir(selection.filename)
    stacks = sampleset.getstack(variables,selection,method='QCD_OSSS',scale=1.1,parallel=parallel) # the 'scale' keyword argument - chosen as 1.1 for mutau - 
                                                                                                   # is an extrapolation factor for the QCD shape from the same-sign
                                                                                                   # to the opposite-sign region
    fname  = "%s/$VAR_%s-%s-%s$TAG"%(outdir,channel,selection.filename,era)
    text   = "%s: %s"%(channel.replace('mu',"#mu").replace('tau',"#tau_{h}"),selection.title)
    for stack, variable in stacks.iteritems():
      outhists.cd(selection.filename)
      for h in stack.hists:
        h.Write(h.GetName().replace("QCD_","QCD"),R.TH1.kOverwrite)
      stack.draw()
      stack.drawlegend(x1=0.6,x2=0.95,y1=0.35,y2=0.95)
      stack.drawtext(text)
      stack.saveas(fname,ext=exts,tag=tag)
      stack.close()
  outhists.Close()
  

def main(args):
  channel = args.channel
  era     = args.era
  parallel = args.parallel
  outdir   = "plots/$ERA"
  histdir  = "hists/$ERA/$CHANNEL.root"
  tag      = ""
  fpattern = args.picopattern

  setera(era) # set era for plot style and lumi-xsec normalization
  sampleset = getsampleset(channel,era,file=fpattern,tag=tag,table=True)
  plot(sampleset,channel,parallel=parallel,tag=tag,outdir=outdir,histdir=histdir,era=era)
  

if __name__ == "__main__":
  import sys
  from argparse import ArgumentParser
  argv = sys.argv
  description = """Simple plotting script for pico analysis tuples"""
  parser = ArgumentParser(prog="plot",description=description,epilog="Good luck!")
  parser.add_argument('-y', '--era',     dest='era', type=str, default='2017',
                                         help="Set era. Default: %(default)s" )
  parser.add_argument('-c', '--channel', dest='channel', type=str, default="mutau",
                                         help="Set channel. Default: %(default)s" )
  parser.add_argument('-s', '--serial',  dest='parallel', action='store_false',
                                         help="Run Tree::MultiDraw serial instead of in parallel" )
  parser.add_argument('-v', '--verbose', dest='verbosity', type=int, nargs='?', const=1, default=0, action='store',
                                         help="Set verbosity" )
  parser.add_argument('--picopattern',   dest='picopattern', type=str, default="$PICODIR/$SAMPLE_$CHANNEL$TAG.root",
                                         help="Name pattern of the flat n-tuple files. Default: %(default)s" )
  args = parser.parse_args()
  LOG.verbosity = args.verbosity
  PLOG.verbosity = args.verbosity
  main(args)
  print "\n>>> Done."
  


