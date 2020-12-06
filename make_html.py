#!/usr/bin/env python3

import uproot
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
use_helvet = True  ## true: use helvetica for plots, make sure the system have the font installed
if use_helvet:
    CMShelvet = hep.style.CMS
    CMShelvet['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.style.use(CMShelvet)
else:
    plt.style.use(hep.style.CMS)

import os
import glob
import shutil
import argparse
parser = argparse.ArgumentParser('create all fit routine')
parser.add_argument('--dir', default=None, help='Path to the base directory of ROOT output.')
parser.add_argument('--bdt', default='900', help='The BDT folder to run. Only in the routine for BDT varying validation we set --bdt 840,860,880,900,920,940')
parser.add_argument('--outweb', default=None, help='Output relative directory to contain the website elements.')
args = parser.parse_args()

if not args.dir:
    raise RuntimeError('--dir is not provided!')
if not args.outweb:
    raise RuntimeError('--outweb is not provided!')


def read_sf_from_log(flist, sf='C'):
    r"""Read the SFs from each of the log file in the given list. Return the list of center, errl, errh values
    """
    if isinstance(flist, str):
        out = [open(flist).readlines()]
    elif isinstance(flist, list):
        out = []
        for f in flist:
            out.append(open(f).readlines())
    
    center, errl, errh = [], [], []
    for content in out:
        for l in content:
            l = l.split()
            if len(l)>0 and l[0]==f'SF_flv{sf}':
                center.append(l[2])
                errl.append(l[3].split('/')[0])
                errh.append(l[3].split('/')[1])
                break
        else:
            center.append('nan')
            errl.append('nan')
            errh.append('nan')
    return center, errl, errh

def fetch_plot(path, outdir_name=None, suffix=''):
    r"""Copy the file (i.e. png, jpg or pdf which are elements to make the web) from the given path to the destination, modify the name with the suffix
    """
    fname = path.split('/')[-1] # filename
    fname = fname[:-4]+suffix+fname[-4:] # filename with suffix
    shutil.copy2(path, os.path.join(outbasedir, outdir_name, fname))
    return fname

def draw_sfbdtvary_plot(wp, pt, ptcut, bdtlist, savepath=None):
    r"""Draw the sfBDT varying plots. [Deprecated in this version]
    """
    ptmin, ptmax = ptcut
    bdtvalues = [int(b[3:])/1000. for b in bdtlist]
    loglist = [os.path.join(indir.replace('_TP_', f'_{wp}_'), 'Cards', b, pt, 'fit.log') for b in bdtlist]
    center, errl, errh = read_sf_from_log(loglist, sf='C')
    print (center, errl, errh)
    f, ax = plt.subplots(figsize=(11,11))
    hep.cms.label(data=True, paper=False, year=2016, ax=ax, rlabel=r'35.9 $fb^{-1}$ (13 TeV)', fontname='sans-serif')
    ax.plot([0,1], [1,1], color='grey', linestyle='dashed')
    ax.errorbar(bdtvalues, center, yerr=[-np.array(errl),np.array(errh)], color='k', marker='s', markersize=8, linestyle='none', label=r'$SF(flvC)\pm unce.(68\%)$')
    ax.fill_between(bdtvalues, np.array(center)+np.array(errl), np.array(center)+np.array(errh), edgecolor='darkblue', facecolor='yellow', linewidth=0) ## draw bkg unce.
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0,2.2); ax.set_xlabel('sfBDT cut value', ha='right', x=1.0)
    ax.text(0.05,0.15,'WP: '+wpname[wp]+r', $p_{T}: '+'({ptmin}, {ptmax})'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+\infty')+r'$ GeV')
    if isinstance(savepath, str):
        plt.savefig(savepath)

def make_new_dir(indir, outbasedir=None):
    r"""Make the new dir to contain all WP (TP, MP, LP) for a given fit variable
    """
    outdir_name = [_ for _ in indir.split('/') if _][-1].replace('_TP_', '_allWP_')
    if not os.path.exists(os.path.join(outbasedir, outdir_name)):
        os.makedirs(os.path.join(outbasedir, outdir_name))
    return outdir_name

def make_website(indir, outdir_name, scanned_wp_list=['TP','MP','LP']):
    r"""The main function to make the website
    """
    print (f'config: {indir}, {outdir_name}')
    bdtlist = os.listdir(os.path.join(indir, 'Cards'))
    if len(user_specified_bdtlist) > 0:
        bdtlist = user_specified_bdtlist
    bdtlist = sorted(bdtlist, key=lambda d: int(d[3:])) # order by bdt cut value
    ptlist = os.listdir(os.path.join(indir, 'Cards', bdtlist[0]))
    ptlist = sorted(ptlist, key=lambda pt: int(pt.split('to')[0].split('pt')[1])) # order by first pt cut value
    ptcutlist = [(int(pt.split('to')[0].split('pt')[1]), int(pt.split('to')[1])) for pt in ptlist]
    print(ptlist)

    wpname = {'TP':'Tight', 'MP':'Medium', 'LP':'Loose'}

    for bdtdir in bdtlist:
        mkdown_str = ''
        # sfBDT range
        mkdown_str += '------------------\n'
        mkdown_str += '# `sfBDT` > %.2f \n' % (int(bdtdir[3:])/1000.)

        ## fit result
        for sf in ['C', 'B', 'L']:
            sf_title = {'C':'cc-tagging SF (`SF_flvC`)', 'B':'bb-mistagging SF (`SF_flvB`)', 'L':'light-mistagging SF (`SF_flvL`)'}
    #     for sf in ['B', 'C', 'L']:
    #         sf_title = {'C':'cc-mistagging SF (`SF_flvC`)', 'B':'bb-tagging SF (`SF_flvB`)', 'L':'light-mistagging SF (`SF_flvL`)'}
            mkdown_str += f'## {sf_title[sf]} \n'
            mkdown_str += '|       | ' + ' | '.join(['pT ({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + ' | \n'
            mkdown_str += '| :---: '*(len(ptlist)+1) + '| \n'
            for wp in scanned_wp_list:
                loglist = [os.path.join(indir.replace('_TP_', f'_{wp}_'), 'Cards', bdtdir, pt, 'fit.log') for pt in ptlist]
                center, errl, errh = read_sf_from_log(loglist, sf=sf) ## sf set to the correct sf!
                mkdown_str += f'| **{wpname[wp]}** WP | ' + ' | '.join([f'**{c}** [{el}/{eh}]' for c, el, eh in zip(center, errl, errh)]) + ' | \n'
                if 'nan' in center+errl+errh:
                    print(f'multifit failed... {bdtdir}')

        if not show_fit_number_only:
            for wp in scanned_wp_list:
                ## pre/post-fit stack
                for prepost in ['prefit', 'postfit']:
                    if prepost=='prefit':
                        mkdown_str += f'## Pre-fit stacked plot (**{wpname[wp]}** WP) \n'
                    elif prepost=='postfit':
                        mkdown_str += f'## Post-fit stacked plot (**{wpname[wp]}** WP) \n'
                    for wpcat in ['pass', 'fail']:
                        # left->right: pT increase
                        mkdown_str += f'### {wpcat}\nIn the order of : pT in ' + ', '.join(['({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + '\n\n'
                        for pt, ptcut in zip(ptlist, ptcutlist):
                            ptmin, ptmax = ptcut
                            plottitle = '({ptmin}, {ptmax}), {wpcat}'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf', wpcat=wpcat)
                            # plot produced from node14, has unique name (deprecated)
                            # plotname  = fetch_plot(os.path.join(plots_massprod_dir, outdir_name.replace('_allWP_', f'_{wp}_') + f'__{bdtdir}__{pt}__{wpcat}.png'))
                            try:
                                plotname  = fetch_plot(os.path.join(indir.replace('_TP_', f'_{wp}_'), 'Cards', bdtdir, pt, f'stack_{prepost}_{wpcat}.png'), outdir_name, suffix=f'__{wp}__{bdtdir}__{pt}')
                                mkdown_str += f'<img src="{plotname}" title="{plottitle}" alt="{plottitle}" style="width:400px;"/> \n'
                            except Exception as e:
                                print(f'pre/post-fit stack failed... {bdtdir}, {wp}, {prepost}, {wpcat}, {pt}', '\n', e)
                                except_str = 'pT ({}, {}) vacant'.format(ptmin, ptmax if ptmax!=100000 else '+inf')
                                mkdown_str += f'<textarea name="a" style="width:400px;height:400px;">{except_str}</textarea> \n'
                        mkdown_str += ' \n'


                ## prefit template
                mkdown_str += f'## Pre/post-fit template (**{wpname[wp]}** WP) \n'
                for wpcat in ['pass', 'fail']:
                    mkdown_str += f'### {wpcat}\nIn the order of : pT in ' + ', '.join(['({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + '\n\n'
                    for pt, ptcut in zip(ptlist, ptcutlist):
                        ptmin, ptmax = ptcut
                        plottitle = '({ptmin}, {ptmax}), {wpcat}'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf', wpcat=wpcat)
                        # named 'pass.png', 'fail.png'
                        try:
                            plotname  = fetch_plot(os.path.join(indir.replace('_TP_', f'_{wp}_'), 'Cards', bdtdir, pt, wpcat+'.png'), outdir_name, suffix=f'__{wp}__{bdtdir}__{pt}')
                            mkdown_str += f'<img src="{plotname}" title="{plottitle}" alt="{plottitle}" style="width:400px;"/> \n'
                        except Exception as e:
                            print(f'prefit template failed... {bdtdir}, {wp}, {wpcat}, {pt}', '\n', e)
                            except_str = 'pT ({}, {}) vacant'.format(ptmin, ptmax if ptmax!=100000 else '+inf')
                            mkdown_str += f'<textarea name="a" style="width:400px;height:400px;">{except_str}</textarea> \n'
                    mkdown_str += ' \n'

                ## unce comp plots
                mkdown_str += f'## Shape unce. variations (**{wpname[wp]}** WP) \n'
                for unce_type in unce_list:
                    mkdown_str += f'### >>> {unce_type} \n'
                    for wpcat in ['pass', 'fail']:
                        mkdown_str += f'### {wpcat}\nIn the order of : pT in ' + ', '.join(['({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + '\n\n'
                        for pt, ptcut in zip(ptlist, ptcutlist):
                            ptmin, ptmax = ptcut
                            plottitle = '({ptmin}, {ptmax}), {wpcat}, {unce_type}'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf', wpcat=wpcat, unce_type=unce_type)
                            # named 'pass.png', 'fail.png'
                            try:
                                plotname  = fetch_plot(os.path.join(indir.replace('_TP_', f'_{wp}_'), 'Cards', bdtdir, pt, f'unce_comp_{unce_type}_{wpcat}.png'), outdir_name, suffix=f'__{wp}__{bdtdir}__{pt}')
                                mkdown_str += f'<img src="{plotname}" title="{plottitle}" alt="{plottitle}" style="width:400px;"/> \n'
                            except Exception as e:
                                print(f'unce comparison plot failed... {bdtdir}, {wp}, {wpcat}, {pt}', '\n', e)
                                except_str = 'pT ({}, {}) vacant'.format(ptmin, ptmax if ptmax!=100000 else '+inf')
                                mkdown_str += f'<textarea name="a" style="width:400px;height:400px;">{except_str}</textarea> \n'
                        mkdown_str += ' \n'

                ## impact plot
                mkdown_str += f'## Impacts plot (**{wpname[wp]}** WP) \n'
                mkdown_str += f'In the order of : pT in ' + ', '.join(['({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + '\n\n'
                for pt, ptcut in zip(ptlist, ptcutlist):
                    ptmin, ptmax = ptcut
                    try:
                        plotname  = fetch_plot(os.path.join(indir.replace('_TP_', f'_{wp}_'), 'Cards', bdtdir, pt, 'impacts.pdf'), outdir_name, suffix=f'__{wp}__{bdtdir}__{pt}')
                        mkdown_str += f'<object data="{plotname}" type="application/pdf" width="700px" height="500px"></object> \n'
                    except Exception as e:
                        print(f'impact plot failed... {bdtdir}, {wp}, {pt}', '\n', e)
                        except_str = 'pT ({}, {}) vacant'.format(ptmin, ptmax if ptmax!=100000 else '+inf')
                        mkdown_str += f'<textarea name="a" style="width:700px;height:500px;">{except_str}</textarea> \n'
                mkdown_str += '\n'

        mkdown_str += '------------------\n'

        # print (mkdown_str)
        with open(os.path.join(outbasedir, outdir_name, bdtdir), 'w') as f:
            f.write(html_template.replace('$TITLE', outdir_name.split('_allWP_')[-1]).replace('$TEXT', mkdown_str))

    if draw_pt_vary:
        mkdown_str = ''
        mkdown_str += '# cc-tagging SF as a function of `sfBDT` cut position \n'
        for wp in scanned_wp_list:
            mkdown_str += f'## **{wpname[wp]}** WP: \nLeft to right: pT in ' + ', '.join(['({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + '\n\n'
            for pt, ptcut in zip(ptlist, ptcutlist):
                ptmin, ptmax = ptcut
                plotname = f'sfbdtvary__{wp}__{pt}.png'
                plottitle = '({ptmin}, {ptmax}), {wpcat}'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf', wpcat=wpcat)
                draw_sfbdtvary_plot(wp, pt, ptcut, bdtlist, savepath=os.path.join(outbasedir, outdir_name, plotname))
                mkdown_str += f'<img src="{plotname}" title="{plottitle}" alt="{plottitle}" style="width:400px;"/> \n'
            mkdown_str += '\n'

        with open(os.path.join(outbasedir, outdir_name, 'bdtvary'), 'w') as f:
            f.write(html_template.replace('$TITLE', outdir_name.split('_allWP_')[-1]).replace('$TEXT', mkdown_str))


html_template = '''<!doctype html>
<html lang="">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <!-- Edit your site info here -->
    <meta name="description" content="EXAMPLE SITE DESCRIPTION">
    <title>$TITLE</title>

    <script src="https://cdn.jsdelivr.net/npm/@webcomponents/webcomponentsjs@2/webcomponents-loader.min.js"></script>
    <script type="module" src="https://cdn.jsdelivr.net/gh/zerodevx/zero-md@1/src/zero-md.min.js"></script>

    <style>
      /* Edit your header styles here */
      header { font-family: sans-serif; font-size: 20px; text-align: center; position: fixed; width: 100%; line-height: 42px; top: 0; left: 0; background-color: #424242; color: white; }
      body { box-sizing: border-box; min-width: 200px; max-width: 2000px; margin: 56px auto 0 auto; padding: 45px; }
      @media (max-width: 767px) {
        header { font-size: 15px; }
        body { padding: 15px; }
      }
    </style>
  </head>
  <body>

  <!-- Edit your Markdown URL file location here -->
  <zero-md>
    <!-- Declare `<template>` element as a child of `<zero-md>` -->
    <template>
      <!-- Wrap your markdown string inside an `<xmp>` tag -->
<xmp>
$TEXT
</xmp>
    </template>
  </zero-md>

    <!-- Edit your header title here -->
    <header class="header">$TITLE</header>

  </body>
</html>'''


## Global vars
unce_list=['pu','fracBB','fracCC','fracLight','sfBDTRwgt','sfBDTFloAround'];
draw_pt_vary = False
show_fit_number_only = False;
outbasedir = args.outweb
user_specified_bdtlist = ['bdt'+bdt_val for bdt_val in args.bdt.split(',')] 

abspath = os.path.abspath(os.getcwd())
for indir in glob.glob(args.dir):
    if '_TP_' in indir:
        make_website(indir, make_new_dir(indir, outbasedir))