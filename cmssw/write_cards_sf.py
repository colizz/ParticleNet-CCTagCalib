import CombineHarvester.CombineTools.ch as ch
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

import argparse
parser = argparse.ArgumentParser('Preprocess ntuples')
parser.add_argument('inputdir',
    help='Input diretory.'
)
args = parser.parse_args()

cb = ch.CombineHarvester()
cb.SetVerbosity(1)

useAutoMCStats = True
inputdir = args.inputdir
outputname = 'SF.txt'

cats = [
    (1, 'pass'),
    (2, 'fail'),
    ]

cb.AddObservations(['*'], [''], ['13TeV'], [''], cats)

bkg_procs = ['flvL']
cb.AddProcesses(['*'], [''], ['13TeV'], [''], bkg_procs, cats, False)

sig_procs = ['flvC', 'flvB']
cb.AddProcesses(['*'], [''], ['13TeV'], [''], sig_procs, cats, True)

all_procs = bkg_procs + sig_procs

bins = cb.bin_set()

shapeSysts = {
    'pu':all_procs,
    'fracBB':['flvB'],
    'fracCC':['flvC'],
    'fracLight':['flvL'],
#     'qcdSyst':all_procs, # deprecated
#     'qcdKdeSyst':all_procs, # deprecated
#     'psWeight':all_procs, # temporarily disabled for test
    'sfBDTRwgt':all_procs,
    'sfBDTFloAround':all_procs,
    }

for syst in shapeSysts:
    cb.cp().process(shapeSysts[syst]).AddSyst(cb, syst, 'shape', ch.SystMap()(1.0))

cb.cp().AddSyst(cb, 'lumi_13TeV', 'lnN', ch.SystMap()(1.025))

# extract shapes from input root files
inputfiles = {bin:[] for bin in bins}
for dp, dn, filenames in os.walk(inputdir):
    if 'ignore' in dp:
        continue
    for f in filenames:
        if f.endswith('.root'):
            bin = f.replace('.root', '').replace('inputs_', '')
            if bin in bins:
                fullpath = os.path.join(dp, f)
                inputfiles[bin].append(fullpath)

for bin in bins:
    cmd = 'hadd -f {bin}.root {inputfiles}'.format(bin=bin, inputfiles=' '.join(inputfiles[bin]))
    print(cmd)
    os.system(cmd)
    cb.cp().bin([bin]).ExtractShapes(
        '%s.root' % bin,
        '$PROCESS',
        '$PROCESS_$SYSTEMATIC'
        )
    os.remove('%s.root' % bin)

if not useAutoMCStats:
    bbb = ch.BinByBinFactory()
    bbb.SetAddThreshold(0.1).SetFixNorm(True)
    bbb.AddBinByBin(cb.cp().backgrounds(), cb)

cb.PrintAll()
 
froot = ROOT.TFile('inputs_%s.root' % outputname.split('.')[0], 'RECREATE')
cb.WriteDatacard(outputname + '.tmp', froot)

with open(outputname, 'w') as fout:
    with open(outputname + '.tmp') as f:
        for l in f:
            if 'rateParam' in l:
                fout.write(l.replace('\n', '  [0.2,5]\n'))
            else:
                fout.write(l)
    os.remove(outputname + '.tmp')

    if useAutoMCStats:
        fout.write('* autoMCStats 20\n')