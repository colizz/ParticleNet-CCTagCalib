#!/usr/bin/env python3

import uproot3 as uproot
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import mplhep as hep
import pandas as pd
use_helvet = True  ## true: use helvetica for plots, make sure the system have the font installed
if use_helvet:
    CMShelvet = hep.style.CMS
    CMShelvet['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.style.use(CMShelvet)
else:
    plt.style.use(hep.style.CMS)

import seaborn as sns
def set_sns_color(*args):
    sns.palplot(sns.color_palette(*args))
    sns.set_palette(*args)

import os
import glob
import argparse
parser = argparse.ArgumentParser('create all fit routine')
parser.add_argument('--dir', default=None, help='Path to the base directory of ROOT output.')
parser.add_argument('--bdt', default='900', help='The BDT folder to run. Set e.g. `--bdt 840,860,880,900,920,940` or `--bdt auto`.')
parser.add_argument('--ext-unce', default=None, help='Extra uncertainty term to run. e.g. --ext-unce NewTerm1,NewTerm2')
parser.add_argument('-t', '--threads', type=int, default=8, help='Concurrent threads to run separate fits.')
args = parser.parse_args()

if not args.dir:
    raise RuntimeError('--dir is not provided!')

import multiprocessing


def make_stacked_plots(inputdir, plot_unce=True, save_plots=True, show_plots=True):
    r"""Make the stacked histograms for both pre-fit and post-fit based on the fitDiagnostics.root
    
    Arguments:
        inputdir: Directory to fitDiagnostics.root
        plot_unce: If or not plot the MC uncertainty in the upper & lower panel. Default: True
        save_plots: If or not store the plot. Default: True
        show_plots: If or not show plot in the runtime. Default: True
    """
    
    year = 2016 if 'SF2016' in inputdir else 2017 if 'SF2017' in inputdir else 2018 if 'SF2018' in inputdir else None
    ## Get the bin info based on inputdir
    for vname, vlabel in bininfo_dm:
        if vname in inputdir:
            break
    else:
        raise RuntimeError('Bininfo not found')
    edges = uproot.open(f'{inputdir}/nominal/inputs_pass.root')['data_obs'].alledges
    if edges[0] == -np.inf:
        edges = edges[1:]
    if edges[-1] == np.inf:
        edges = edges[:-1]
    xmin, xmax, nbin = min(edges), max(edges), len(edges)
    
    print(inputdir, '--stacked--')
    
    ## All information read from fitDiagnostics.root
    fit = uproot.open(f'{inputdir}/fitDiagnostics.root')
    for rootdir, title in zip(['shapes_prefit', 'shapes_fit_s'], ['prefit', 'postfit']):
        for b in ['pass', 'fail']:
            set_sns_color(color_order)
            f = plt.figure(figsize=(12,12))
            gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05) 
            
            ## Upper histogram panel
            ax = f.add_subplot(gs[0])
            hep.cms.label(data=True, paper=False, year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)'%lumi[year], fontname='sans-serif')
            ax.set_xlim(xmin, xmax); ax.set_xticklabels([]); 
            ax.set_ylabel('Events / bin', ha='right', y=1.0)
            label, hdm = {}, {}
            underflow = False if vlabel[-2:] in ['-u','-a'] else True
            overflow  = False if vlabel[-2:] in ['-o','-a'] else True
            if vlabel[-2:] in ['-u','-o','-a']:
                vlabel = vlabel[:-2]

            content = [fit[f'{rootdir}/{b}/{cat}'].allvalues[1:-1] for cat in cat_order]
            hep.histplot(content, bins=edges, label=[f'MC ({cat})' for cat in cat_order], histtype='fill', edgecolor='k', linewidth=1, stack=True) ## draw MC
            bkgtot, bkgtot_err = fit[f'{rootdir}/{b}/total'].allvalues[1:-1], np.sqrt(fit[f'{rootdir}/{b}/total'].allvariances[1:-1])
            if plot_unce:
                ax.fill_between(edges, (bkgtot-bkgtot_err).tolist()+[0], (bkgtot+bkgtot_err).tolist()+[0], label='BKG total unce.', step='post', hatch='\\\\', edgecolor='dimgrey', facecolor='none', linewidth=0) ## draw bkg unce.
            data, data_errh, data_errl = fit[f'{rootdir}/{b}/data'].yvalues, fit[f'{rootdir}/{b}/data'].yerrorshigh, fit[f'{rootdir}/{b}/data'].yerrorslow
            hep.histplot(data, yerr=(data_errl, data_errh), bins=edges, label='Data', histtype='errorbar', color='k', markersize=15, elinewidth=1.5) ## draw data
            ax.set_ylim(0, ax.get_ylim()[1])
            if plot_unce:
                ax.set_ylim(0, 1.8*max(data))
            ax.legend()

            ## Ratio panel
            ax1 = f.add_subplot(gs[1]); ax1.set_xlim(xmin, xmax); ax1.set_ylim(0.001, 1.999)
            ax1.set_xlabel(vlabel, ha='right', x=1.0); ax1.set_ylabel('Data / MC', ha='center')
            ax1.plot([xmin,xmax], [1,1], 'k'); ax1.plot([xmin,xmax], [0.5,0.5], 'k:'); ax1.plot([xmin,xmax], [1.5,1.5], 'k:')

            if plot_unce:
                ax1.fill_between(edges, ((bkgtot-bkgtot_err)/bkgtot).tolist()+[0], ((bkgtot+bkgtot_err)/bkgtot).tolist()+[0], step='post', hatch='\\\\', edgecolor='dimgrey', facecolor='none', linewidth=0) ## draw bkg unce.
            hep.histplot(data/bkgtot, yerr=(data_errl/bkgtot, data_errh/bkgtot), bins=edges, histtype='errorbar', color='k', markersize=15, elinewidth=1)

            plot_unce_suf = '' if plot_unce else 'noUnce'
            
            if save_plots:
                plt.savefig(f'{inputdir}/stack_{title}_{b}{plot_unce_suf}.png')
                plt.savefig(f'{inputdir}/stack_{title}_{b}{plot_unce_suf}.pdf')
                if not show_plots:
                    plt.close()


def make_stacked_plots_for_shapeunc(inputdir, unce_type=None, plot_unce=True, draw_stacked_plots=False, save_unce_comp_plots=True, show_plots=True, norm_unce=False):
    r"""Make the shape comparison and/or the stacked histograms for a specific type of shape uncertainty based on the fitDiagnostics.root
    
    Arguments:
        inputdir: Directory to fitDiagnostics.root
        unce_type: Name of shape uncertainty (w/o Up or Down) to plot.
        plot_unce: If or not plot the MC uncertainty in the upper & lower panel. Default: True
        draw_stacked_plots: If or not also draw the stacked histograms (drawing the comparison plots is the default option). Default: False
        save_unce_comp_plots: If or not store the shape comparison plot. Default: True
        show_plots: If or not show plot in the runtime. Default: True
        norm_unce: Normalize the up/down uncertainty to nominal. Default: False
    """

    year = 2016 if 'SF2016' in inputdir else 2017 if 'SF2017' in inputdir else 2018 if 'SF2018' in inputdir else None
    for vname, vlabel in bininfo_dm:
        if vname in inputdir:
            break
    else:
        raise RuntimeError('Bininfo not found')
    
    import os
    if not isinstance(unce_type, str) or not os.path.exists(f'{inputdir}/{unce_type}Up') or not os.path.exists(f'{inputdir}/{unce_type}Down'):
        raise RuntimeError('Uncertainty type not exist')
    edges = uproot.open(f'{inputdir}/nominal/inputs_pass.root')['data_obs'].alledges
    if edges[0] == -np.inf:
        edges = edges[1:]
    if edges[-1] == np.inf:
        edges = edges[:-1]
    xmin, xmax, nbin = min(edges), max(edges), len(edges)

    print(inputdir, '--unce--', unce_type)
    
    # curves for unce
    for b in ['pass', 'fail']:
        content = [uproot.open(f'{inputdir}/nominal/inputs_{b}.root')[f'{cat}'].allvalues[1:-1] for cat in cat_order[::-1]]
        yerror  = [np.sqrt(uproot.open(f'{inputdir}/nominal/inputs_{b}.root')[f'{cat}'].allvariances[1:-1]) for cat in cat_order[::-1]]
        content_up   = [uproot.open(f'{inputdir}/{unce_type}Up/inputs_{b}.root')[f'{cat}_{unce_type}Up'].allvalues[1:-1] for cat in cat_order[::-1]]
        content_down = [uproot.open(f'{inputdir}/{unce_type}Down/inputs_{b}.root')[f'{cat}_{unce_type}Down'].allvalues[1:-1] for cat in cat_order[::-1]]
        lab_suf = ''
        if norm_unce:
            lab_suf = '(norm)'
            for icat, cat in enumerate(cat_order[::-1]):
                content_up[icat] *= content[icat].sum() / content_up[icat].sum()
                content_down[icat] *= content[icat].sum() / content_down[icat].sum()
        f, ax = plt.subplots(figsize=(12,12))
        hep.cms.label(data=True, paper=False, year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)'%lumi[year], fontname='sans-serif')
        for icat, (cat, color) in enumerate(zip(cat_order[::-1], ['blue', 'red', 'green'])):
            hep.histplot(content[icat], yerr=yerror[icat], bins=edges, label=f'MC ({cat})', color=color)
        for icat, (cat, color) in enumerate(zip(cat_order[::-1], ['blue', 'red', 'green'])):
            hep.histplot(content_up[icat], bins=edges, label=f'MC ({cat}) {unce_type}Up {lab_suf}', color=color, linestyle='--')
        for icat, (cat, color) in enumerate(zip(cat_order[::-1], ['blue', 'red', 'green'])):
            hep.histplot(content_down[icat], bins=edges, label=f'MC ({cat}) {unce_type}Down {lab_suf}', color=color, linestyle=':')
        ax.set_xlim(xmin, xmax); ax.set_xlabel(vlabel, ha='right', x=1.0); ax.set_ylabel('Events / bin', ha='right', y=1.0)
        ax.legend(prop={'size': 18})
        
        if save_unce_comp_plots:
            plt.savefig(f'{inputdir}/unce_comp_{unce_type}_{b}.png')
            plt.savefig(f'{inputdir}/unce_comp_{unce_type}_{b}.pdf')
            if not show_plots:
                plt.close()

    # stacked plots
    if draw_stacked_plots:
        for filedir in ['nominal', unce_type+'Up', unce_type+'Down']:
            roothist_suf = '' if filedir=='nominal' else '_'+filedir
            for b in ['pass', 'fail']:
                set_sns_color(color_order)
                f = plt.figure(figsize=(12,12))
                gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05) 
                ax = f.add_subplot(gs[0])
                hep.cms.label(data=True, paper=False, year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)'%lumi[year], fontname='sans-serif')
                ax.set_xlim(xmin, xmax); ax.set_xticklabels([]); 
                ax.set_ylabel('Events / bin', ha='right', y=1.0)
                label, hdm = {}, {}
                underflow = False if vlabel[-2:] in ['-u','-a'] else True
                overflow  = False if vlabel[-2:] in ['-o','-a'] else True
                if vlabel[-2:] in ['-u','-o','-a']:
                    vlabel = vlabel[:-2]

                content = [uproot.open(f'{inputdir}/{filedir}/inputs_{b}.root')[f'{cat}{roothist_suf}'].allvalues[1:-1] for cat in cat_order]
                bkgtot = np.sum(content, axis=0)
                hep.histplot(content, bins=edges, label=[f'MC ({cat})' for cat in cat_order], histtype='fill', edgecolor='k', linewidth=1, stack=True) ## draw MC
                data = uproot.open(f'{inputdir}/nominal/inputs_{b}.root')['data_obs'].allvalues[1:-1]
                data_errh = data_errl = np.sqrt(uproot.open(f'{inputdir}/nominal/inputs_{b}.root')['data_obs'].allvariances[1:-1])
                hep.histplot(data, yerr=(data_errl, data_errh), bins=edges, label='Data', histtype='errorbar', color='k', markersize=15, elinewidth=1.5) ## draw data
                ax.set_ylim(0, ax.get_ylim()[1])
                ax.legend()

                ax1 = f.add_subplot(gs[1]); ax1.set_xlim(xmin, xmax); ax1.set_ylim(0.001, 1.999)
                ax1.set_xlabel(vlabel, ha='right', x=1.0); ax1.set_ylabel('Data / MC', ha='center')
                ax1.plot([xmin,xmax], [1,1], 'k'); ax1.plot([xmin,xmax], [0.5,0.5], 'k:'); ax1.plot([xmin,xmax], [1.5,1.5], 'k:')

                hep.histplot(data/bkgtot, yerr=(data_errl/bkgtot, data_errh/bkgtot), bins=edges, histtype='errorbar', color='k', markersize=15, elinewidth=1)
                
                if not show_plots:
                    plt.close()


def make_plots_wrapper(inputdir, unce_list):
    r"""Launch the plot maker for a single fit point.
    
    Arguments:
        inputdir: Directory to fitDiagnostics.root
        unce_list: List of shape uncertainties to draw the uncertainty plot
    """
    print ('- launch:', inputdir)
    mpl.use('AGG') # no rendering plots in the window
    try:
        make_stacked_plots(inputdir, show_plots=False)
        for unce_type in unce_list:
            make_stacked_plots_for_shapeunc(inputdir, unce_type=unce_type, show_plots=False)
    except Exception as e:
        print('inputdir failed:', inputdir, 'Error:', e)


import yaml
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'config.yml')) as f:
    config = yaml.safe_load(f)

if config['tagger']['type'].lower() == 'cc':
    flv1, flv2 = 'C', 'B'
elif config['tagger']['type'].lower() == 'bb':
    flv1, flv2 = 'B', 'C'
else:
    raise RuntimeError('Tagger type in config.yml must be cc or bb.')

lumi = {2016: 35.92, 2017: 41.53, 2018: 59.74}
bininfo_dm = [ # vtitle_contains, vlabel
    ("csvv2", r'$CSVv2$'),
    ('msv12_ptmax_log', r'$log(m_{SV1,p_{T}\,max}\; /GeV)$'),
    ('msv12_dxysig_log', r'$log(m_{SV1,d_{xy}sig\,max}\; /GeV)$'),
]
color_order, cat_order = sns.color_palette('cubehelix_r', 3), ['flvL', f'flv{flv2}', f'flv{flv1}']
unce_list = ['pu','fracBB','fracCC','fracLight','psWeightIsr','psWeightFsr','sfBDTRwgt']
if args.ext_unce is not None:
    unce_list += args.ext_unce.split(',')

from cmssw.utils import find_valid_runlist

pool = multiprocessing.Pool(processes=args.threads)
for inputdir in find_valid_runlist(args.dir, bdt_mode=args.bdt):
    ## Submit a multiprocess job
    pool.apply_async(make_plots_wrapper, args=(inputdir, unce_list,))
            
pool.close()
pool.join()