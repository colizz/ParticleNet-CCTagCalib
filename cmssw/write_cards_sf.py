import CombineHarvester.CombineTools.ch as ch
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

import argparse
parser = argparse.ArgumentParser('Preprocess ntuples')
parser.add_argument('inputdir', help='Input diretory.')
parser.add_argument('--ext-unce', default=None, help='Extra uncertainty term to run. e.g. --ext-unce NewTerm1,NewTerm2')
parser.add_argument('--run-impact', action='store_true', help='Run impact plots.')
parser.add_argument('--run-unce-breakdown', action='store_true', help='Run uncertainty breakdown')
args = parser.parse_args()

import yaml
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, '../config.yml')) as f:
    config = yaml.safe_load(f)

if config['tagger']['type'].lower() == 'cc':
    flv_poi1, flv_poi2 = 'flvC', 'flvB'
elif config['tagger']['type'].lower() == 'bb':
    flv_poi1, flv_poi2 = 'flvB', 'flvC'
else:
    raise RuntimeError('Tagger type in config.yml must be cc or bb.')

import subprocess
def runcmd(cmd):
    """Run a shell command"""
    p = subprocess.Popen(
        cmd, shell=True, universal_newlines=True
    )
    out, _ = p.communicate()
    return (out, p.returncode)

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

sig_procs = [flv_poi1, flv_poi2]
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
    'psWeightIsr':all_procs,
    'psWeightFsr':all_procs,
    'sfBDTRwgt':all_procs,
#     'sfBDTFloAround':all_procs, # deprecated since v2
#     'fitVarRwgt':all_procs,
    }
if args.ext_unce is not None:
    for ext_unce in args.ext_unce.split(','):
        shapeSysts[ext_unce] = all_procs

for syst in shapeSysts:
    cb.cp().process(shapeSysts[syst]).AddSyst(cb, syst, 'shape', ch.SystMap()(1.0))

cb.cp().AddSyst(cb, 'lumi_13TeV', 'lnN', ch.SystMap()(1.025))

# extract shapes from input root files
inputfiles = {bin:[] for bin in bins}
for dp, dn, filenames in os.walk(inputdir):
    if 'ignore' in dp:
        continue
    for f in filenames:
        if f in ['inputs_pass.root', 'inputs_fail.root']:
            bin = f.replace('.root', '').replace('inputs_', '')
            if bin in bins:
                fullpath = os.path.join(dp, f)
                inputfiles[bin].append(fullpath)

for bin in bins:
    cmd = 'hadd -f {inputdir}/{bin}.root {inputfiles}'.format(inputdir=args.inputdir, bin=bin, inputfiles=' '.join(inputfiles[bin]))
    print(cmd)
    runcmd(cmd)
    cb.cp().bin([bin]).ExtractShapes(
        os.path.join(args.inputdir, '%s.root' % bin),
        '$PROCESS',
        '$PROCESS_$SYSTEMATIC'
        )
    os.remove(os.path.join(args.inputdir, '%s.root' % bin))

if not useAutoMCStats:
    bbb = ch.BinByBinFactory()
    bbb.SetAddThreshold(0.1).SetFixNorm(True)
    bbb.AddBinByBin(cb.cp().backgrounds(), cb)

cb.PrintAll()

froot = ROOT.TFile(os.path.join(args.inputdir, 'inputs_%s.root' % outputname.split('.')[0]), 'RECREATE')
cb.WriteDatacard(os.path.join(args.inputdir, outputname + '.tmp'), froot)
froot.Close()

with open(os.path.join(args.inputdir, outputname), 'w') as fout:
    with open(os.path.join(args.inputdir, outputname + '.tmp')) as f:
        for l in f:
            if 'rateParam' in l:
                fout.write(l.replace('\n', '  [0.2,5]\n'))
            else:
                fout.write(l)
    os.remove(os.path.join(args.inputdir, outputname + '.tmp'))

    if useAutoMCStats:
        fout.write('* autoMCStats 20\n')

## Start running higgs combine

runcmd('''
cd {inputdir} && \
echo "+++ Converting datacard to workspace +++" && \
text2workspace.py -m 125 -P HiggsAnalysis.CombinedLimit.TagAndProbeExtendedV2:tagAndProbe SF.txt --PO categories={flv_poi1},{flv_poi2},flvL && \
echo "+++ Fitting... +++" && \
combine -M MultiDimFit -m 125 SF.root --algo=singles --robustFit=1 | tee fit.log && \
combine -M FitDiagnostics -m 125 SF.root --saveShapes --saveWithUncertainties --robustFit=1 > /dev/null 2>&1
'''.format(inputdir=args.inputdir, flv_poi1=flv_poi1, flv_poi2=flv_poi2)
)

if args.run_impact:
    runcmd('''
cd {inputdir} && \
combineTool.py -M Impacts -d SF.root -m 125 --doInitialFit --robustFit 1 >> pdf.log 2>&1 && \
combineTool.py -M Impacts -d SF.root -m 125 --robustFit 1 --doFits >> pdf.log 2>&1 && \
combineTool.py -M Impacts -d SF.root -m 125 -o impacts.json >> pdf.log 2>&1 && \
plotImpacts.py -i impacts.json -o impacts >> pdf.log 2>&1
'''.format(inputdir=args.inputdir)
    )

if args.run_unce_breakdown:
    runcmd('''
cd {inputdir} && \
combine -M MultiDimFit -m 125 SF.root --algo=grid --robustFit 1 --points 50 -n Grid --redefineSignalPOIs SF_{flv_poi1} && \
plot1DScan.py higgsCombineGrid.MultiDimFit.mH125.root --POI SF_{flv_poi1} && \
combine -M MultiDimFit -m 125 SF.root --algo=singles --robustFit=1 -n Bestfit --saveWorkspace && \
combine -M MultiDimFit -m 125 --algo=grid --points 50  -n Stat higgsCombineBestfit.MultiDimFit.mH125.root --redefineSignalPOIs SF_{flv_poi1} --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances && \
plot1DScan.py higgsCombineGrid.MultiDimFit.mH125.root --others 'higgsCombineStat.MultiDimFit.mH125.root:FreezeAll:2' --POI SF_{flv_poi1} -o unce_breakdown --breakdown Syst,Stat
'''.format(inputdir=args.inputdir, flv_poi1=flv_poi1)
    )
