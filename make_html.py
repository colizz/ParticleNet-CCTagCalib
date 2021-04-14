#!/usr/bin/env python3

import os
import glob
import shutil
import numpy as np
import argparse
parser = argparse.ArgumentParser('create all fit routine')
parser.add_argument('--cfg', default='config.yml', help='YAML config to get global vars.')
parser.add_argument('--dir', default=None, help='Path to the base directory of ROOT output.')
parser.add_argument('--outweb', default=None, help='Output relative directory to contain the website elements.')
parser.add_argument('--bdt', default='900', help='The BDT folder to run. Set to a single value or use `--bdt auto`.')
parser.add_argument('--ext-unce', default=None, help='Extra uncertainty term to run or term to be excluded. e.g. --ext-unce NewTerm1,NewTerm2,~ExcludeTerm1')
parser.add_argument('--show-fit-number-only', action='store_true', help='Only summerise the fit result without copying all the plots to the web folder.')
parser.add_argument('--draw-sfbdt-vary', action='store_true', help='Make the plots for SF as a function of sfBDT cut value.')
parser.add_argument('--draw-sfbdt-vary-dryrun', action='store_true', help='Make the plots for SF as a function of sfBDT cut value (dry run -- do not make new plots, but organize the webpage as if the plots exist).')
parser.add_argument('--draw-sfbdt-vary-set-ymax', default=1.8, help='Set the y-max of the sfBDT variation plot for the *main* SF.')
parser.add_argument('--draw-sfbdt-vary-with-bl', action='store_true', help='Use together with --draw-sfbdt-vary(-dryrun). Also draw SF_b and SF_light plots as a function of sfBDT cut value.')
parser.add_argument('--show-unce-breakdown', action='store_true', help='Show the uncertainty breakdown plots')
parser.add_argument('--show-fitvarrwgt-unce', action='store_true', help='Show the extra uncertainty in the table obtained from an alternative reweighting on the fit variable')
parser.add_argument('--combine-bdtmod', action='store_true', help='In the calculation of err(max d), include the sfBDT variation fit results in the modified BDT cut scheme')
args = parser.parse_args()

if not args.dir:
    raise RuntimeError('--dir is not provided!')
if not args.outweb:
    raise RuntimeError('--outweb is not provided!')

import sys, re
if re.search('CMSSW_\d+_\d+_\d+', os.path.dirname(sys.executable)):
    raise RuntimeError('Please use the miniconda env to run this python command.')
                
def get_dir(indir, wp, bdtdir, ptdir):
    assert f'_{wp0}_' in indir
    import glob
    outdir_list = []
    outdir_list += glob.glob(os.path.join(indir.replace(f'_{wp0}_', f'_{wp}_'), 'Cards', bdtdir, ptdir))
    outdir_list += glob.glob(os.path.join(indir.replace(f'_{wp0}_', f'_{wp}_'), 'Cards', ptdir, bdtdir))
    if len(outdir_list) == 0:
        raise RuntimeError('Directory does not exist: ', (indir, wp, bdtdir, ptdir))
    elif len(outdir_list) == 1:
        return outdir_list[0]
    else:
        return outdir_list

def read_sf_from_log(flist, sf):
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

def fetch_plot(path, outdir=None, suffix=''):
    r"""Copy the file (i.e. png, jpg or pdf which are elements to make the web) from the given path to the destination, modify the name with the suffix
    """
    fname = path.split('/')[-1] # filename
    fname = fname[:-4]+suffix+fname[-4:] # filename with suffix
    shutil.copy2(path, os.path.join(outdir, fname))
    return fname

def draw_sfbdtvary_plot(indir, wp, pt, ptcut, bdtlist, sf, savepath=None):
    r"""Draw the sfBDT varying plots.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    import mplhep as hep
    use_helvet = True  ## true: use helvetica for plots, make sure the system have the font installed
    if use_helvet:
        CMShelvet = hep.style.CMS
        CMShelvet['font.sans-serif'] = ['Helvetica', 'Arial']
        plt.style.use(CMShelvet)
    else:
        plt.style.use(hep.style.CMS)
    
    import matplotlib as mpl
    mpl.use('AGG') # no rendering plots in the window
    
    if sf == flv1:
        ymax, facecolor = float(args.draw_sfbdt_vary_set_ymax), 'yellow'
    elif sf == flv2:
        ymax, facecolor = 3.0, 'greenyellow'
    elif sf == 'L':
        ymax, facecolor = 3.0, 'skyblue'

    def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r', edgecolor=None, alpha=0.5, label=None):
        # Create list for all the error patches
        errorboxes = []
        # Loop over data points; create box from errors at each point
        for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
            rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
            errorboxes.append(rect)
        # Create patch collection with specified colour/alpha
        pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                             edgecolor=edgecolor)
        # Add collection to axes
        ax.add_collection(pc)
        # Plot errorbars
        artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror, fmt='None', ecolor=facecolor, label=label)
        return artists
    
    year = 2016 if 'SF2016' in indir else 2017 if 'SF2017' in indir else 2018 if 'SF2018' in indir else None
    assert year is not None
    
    # Read SF results via the sfBDT list
    ptmin, ptmax = ptcut
    bdtvalues = [int(b[3:])/1000. for b in bdtlist]
    loglist = [os.path.join(get_dir(indir, wp, b, pt), 'fit.log') for b in bdtlist]
    center, errl, errh = read_sf_from_log(loglist, sf=sf)
    center, errl, errh = list(map(float, center)), list(map(float, errl)), list(map(float, errh))
    
    # Plot SF points with errorbars
    print('Producing varied sfBDT plots for', wp, pt, ptcut, '...')
    f, ax = plt.subplots(figsize=(11,11))
    hep.cms.label(data=True, paper=False, year=year, ax=ax, rlabel=r'%s $fb^{-1}$ (13 TeV)'%lumi[year], fontname='sans-serif')
    ax.plot([0,1], [1,1], color='grey', linestyle='dashed')
    ax.errorbar(bdtvalues, center, yerr=[-np.array(errl),np.array(errh)], color='k', marker='s', markersize=8, linestyle='none', label=r'$SF(flv%s)\pm unce.$' % sf)
    ax.fill_between(bdtvalues, np.array(center)+np.array(errl), np.array(center)+np.array(errh), edgecolor='darkblue', facecolor=facecolor, linewidth=0) ## draw bkg unce.
    
    # Plot central box
    cidx = int(len(bdtlist)/2)
    make_error_boxes(ax, xdata=[bdtvalues[cidx]], ydata=[center[cidx]],
                     xerror=np.array([[(bdtvalues[cidx]-bdtvalues[cidx-1])/4],[(bdtvalues[cidx+1]-bdtvalues[cidx])/4]]),
                     yerror=np.array([[-errl[cidx]], [errh[cidx]]]), 
                     alpha=0.2, facecolor='red', label=r'$SF(flv%s)\pm unce.$ (central)' % sf) 
    ax.legend()
    margin = bdtvalues[1] - bdtvalues[0]
    ax.set_xlim(min(bdtvalues)-margin, min(max(bdtvalues)+margin, 1)); ax.set_ylim(0, ymax); ax.set_xlabel('sfBDT cut value', ha='right', x=1.0)
    ax.text(0.10,0.10,wpname[wp], fontweight='bold', transform=ax.transAxes)
    ax.text(0.45,0.10,'$p_{T}: '+'[{ptmin}, {ptmax})'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '\infty')+r'$ GeV', transform=ax.transAxes)
    if isinstance(savepath, str):
        plt.savefig(savepath)
        plt.savefig(savepath.replace('.png','.pdf'))
        plt.close()

def make_new_dir(indir_name, outbasedir=None):
    r"""Make the new dir to contain all WPs e.g. {HP, MP, LP} for a given fit variable
    """
    outdir_name = indir_name.replace(f'_{wp0}_', '_allWP_')
    if not os.path.exists(os.path.join(outbasedir, outdir_name)):
        os.makedirs(os.path.join(outbasedir, outdir_name))
    return os.path.join(outbasedir, outdir_name)

def make_website(indir, basedir, outdir, ptlist, user_pt_bdt_map, user_pt_bdtvarylist_map, scanned_wp_list):
    r"""The main function to make the website
    """
    print (f'config:\n - indir: {indir}\n - outdir: {outdir}\n - ptlist: {ptlist}\n - user_pt_bdt_map: {user_pt_bdt_map}\n - user_pt_bdtvarylist_map: {user_pt_bdtvarylist_map}')
    ptcutlist = [(int(pt.split('to')[0].split('pt')[1]), int(pt.split('to')[1])) for pt in ptlist]
    
    mkdown_str = ''
    # sfBDT range
    mkdown_str += '------------------\n'
    mkdown_str += '# Central `sfBDT` WP \n'
    mkdown_str += f'### `sfBDT` cut set as: pT = ' \
                  + ', '.join(['({}, {}): >{:.3f}'.format(ptmin, ptmax if ptmax!=100000 else '+inf', int(user_pt_bdt_map[f'pt{ptmin}to{ptmax}'][3:])/1000.) for ptmin, ptmax in ptcutlist]) + '\n\n'

    ## fit result
    sf_list = [flv1, flv2, 'L']
    if flv1 == 'C':
        sf_title = {'C':'cc-tagging SF (`SF_flvC`)', 'B':'bb-mistagging SF (`SF_flvB`)', 'L':'light-mistagging SF (`SF_flvL`)'}
    elif flv1 == 'B':
        sf_title = {'B':'bb-tagging SF (`SF_flvB`)', 'C':'cc-mistagging SF (`SF_flvC`)', 'L':'light-mistagging SF (`SF_flvL`)'}
    for sf in sf_list:
        mkdown_str += f'## {sf_title[sf]} \n'
        mkdown_str += '|       | ' + ' | '.join(['pT ({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + ' | \n'
        mkdown_str += '| :---: '*(len(ptlist)+1) + '| \n'
        for wp in scanned_wp_list:
            loglist = [os.path.join(get_dir(indir, wp, user_pt_bdt_map[pt], pt), 'fit.log') for pt in ptlist]
            center, errl, errh = read_sf_from_log(loglist, sf=sf) ## sf set to the correct sf!
            mkdown_str += f'| **{wpname[wp]}** WP | ' + ' | '.join([f'**{c}** [{el}/{eh}]' for c, el, eh in zip(center, errl, errh)]) + ' | \n'
            if 'nan' in center+errl+errh:
                print(f'multifit failed... ', wp, 'central SFs: ', center)
        ## extra uncertainty on BDT variation
        if args.bdt == 'auto' and (args.draw_sfbdt_vary or args.draw_sfbdt_vary_dryrun) and sf == sf_list[0]:
            mkdown_str += f'## {sf_title[sf]} (after external correction from BDT variation and/or fit variable reweigting) \n'
            if args.combine_bdtmod:
                mkdown_str += '(*Note: in the calculation of err(max d), sfBDT variation fit results in the modified BDT cut scheme are also included.) \n\n'
            mkdown_str += '|       | ' + ' | '.join(['pT ({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + ' | \n'
            mkdown_str += '| :---: '*(len(ptlist)+1) + '| \n'
            for wp in scanned_wp_list:
                center_ptlist, errl_ptlist, errh_ptlist, maxdist_ptlist = [], [], [], []
                errl0_ptlist, errh0_ptlist = [], []
                err_fvr_ptlist = []
                for pt in ptlist:
                    loglist = [os.path.join(d, 'fit.log') for d in get_dir(indir, wp, 'bdt*', pt)]  ## get all logs for BDT variation
                    assert isinstance(loglist, list)
                    center, _, _ = read_sf_from_log(loglist, sf=sf)
                    center = np.array([float(c) for c in center if c != 'nan'])
                    if args.combine_bdtmod:
                        indir = indir[:-1] if indir[-1]=='/' else indir
                        indir_fvr ='/'.join([indir.rsplit('/', 1)[0], 'bdtmod', indir.rsplit('/', 1)[1]]) # the indir path for bdtmod
                        loglist = [os.path.join(d, 'fit.log') for d in get_dir(indir_fvr, wp, 'bdt*', pt)]  ## get all logs for BDT variation in the path of bdtmod
                        center_mod, _, _ = read_sf_from_log(loglist, sf=sf)
                        center_mod = np.array([float(c) for c in center_mod if c != 'nan'])
                        center = np.concatenate([center, center_mod])  # concat results
                    center0, errl0, errh0 = read_sf_from_log([os.path.join(get_dir(indir, wp, user_pt_bdt_map[pt], pt), 'fit.log')], sf=sf)
                    center0, errl0, errh0 = float(center0[0]), float(errl0[0]), float(errh0[0])
                    if args.show_fitvarrwgt_unce:
                        indir = indir[:-1] if indir[-1]=='/' else indir
                        indir_fvr ='/'.join([indir.rsplit('/', 1)[0], 'fitvarrwgt', indir.rsplit('/', 1)[1]]) # the indir path for fitvarrwgt
                        center0_fvr, _, _ = read_sf_from_log([os.path.join(get_dir(indir_fvr, wp, user_pt_bdt_map[pt], pt), 'fit.log')], sf=sf)
                        center0_fvr = float(center0_fvr[0])
                        err_fvr = np.abs(center0_fvr - center0)
                    import math
                    if not math.isnan(center0):
                        maxdist = np.max(np.abs(center - center0))
                        center_ptlist.append('+%.3f' % center0)
                        errl0_ptlist.append('%.3f' % errl0)
                        errh0_ptlist.append('%.3f' % errh0)
                        maxdist_ptlist.append('%.3f' % maxdist)
                        if args.show_fitvarrwgt_unce:
                            err_fvr_ptlist.append('%.3f' % err_fvr)
                            errl_ptlist.append('%.3f' % -np.sqrt(errl0**2 + maxdist**2 + err_fvr**2))
                            errh_ptlist.append('+%.3f' % np.sqrt(errh0**2 + maxdist**2 + err_fvr**2))
                        else:
                            errl_ptlist.append('%.3f' % -np.sqrt(errl0**2 + maxdist**2))
                            errh_ptlist.append('+%.3f' % np.sqrt(errh0**2 + maxdist**2))
                    else:
                        center_ptlist.append('nan'); errl0_ptlist.append('nan'); errh0_ptlist.append('nan'); maxdist_ptlist.append('nan')
                        err_fvr_ptlist.append('nan'); errl_ptlist.append('nan'); errh_ptlist.append('nan')
                if args.show_fitvarrwgt_unce:
                    mkdown_str += f'| **{wpname[wp]}** WP | ' + \
                                  ' | '.join([f'**{c}** ([{el}/{eh}]<sub>orig</sub> [±{md}]<sub>max d</sub> [±{efvr}]<sub>rwgt msv</sub>)</br>[{eltot}/{ehtot}]<sub>tot</sub>' for c, el, eh, md, efvr, eltot, ehtot in \
                                             zip(center_ptlist, errl0_ptlist, errh0_ptlist, maxdist_ptlist, err_fvr_ptlist, errl_ptlist, errh_ptlist)]) + \
                                  ' | \n'
                else:
                    mkdown_str += f'| **{wpname[wp]}** WP | ' + \
                                  ' | '.join([f'**{c}** ([{el}/{eh}]<sub>orig</sub> [±{md}]<sub>max d</sub>)</br>[{eltot}/{ehtot}]<sub>tot</sub>' for c, el, eh, md, eltot, ehtot in \
                                             zip(center_ptlist, errl0_ptlist, errh0_ptlist, maxdist_ptlist, errl_ptlist, errh_ptlist)]) + \
                                  ' | \n'

    if not args.show_fit_number_only:
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
                        bdtdir = user_pt_bdt_map[pt]
                        plottitle = '({ptmin}, {ptmax}), {wpcat}'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf', wpcat=wpcat)
                        # plot produced from node14, has unique name (deprecated)
                        # plotname  = fetch_plot(os.path.join(plots_massprod_dir, outdir.replace('_allWP_', f'_{wp}_') + f'__{bdtdir}__{pt}__{wpcat}.png'))
                        try:
                            plotname  = fetch_plot(os.path.join(get_dir(indir, wp, bdtdir, pt), f'stack_{prepost}_{wpcat}.png'), outdir, suffix=f'__{wp}__{bdtdir}__{pt}')
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
                    bdtdir = user_pt_bdt_map[pt]
                    plottitle = '({ptmin}, {ptmax}), {wpcat}'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf', wpcat=wpcat)
                    # named 'pass.png', 'fail.png'
                    try:
                        plotname  = fetch_plot(os.path.join(get_dir(indir, wp, bdtdir, pt), wpcat+'.png'), outdir, suffix=f'__{wp}__{bdtdir}__{pt}')
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
                        bdtdir = user_pt_bdt_map[pt]
                        plottitle = '({ptmin}, {ptmax}), {wpcat}, {unce_type}'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf', wpcat=wpcat, unce_type=unce_type)
                        # named 'pass.png', 'fail.png'
                        try:
                            plotname  = fetch_plot(os.path.join(get_dir(indir, wp, bdtdir, pt), f'unce_comp_{unce_type}_{wpcat}.png'), outdir, suffix=f'__{wp}__{bdtdir}__{pt}')
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
                bdtdir = user_pt_bdt_map[pt]
                try:
                    plotname  = fetch_plot(os.path.join(get_dir(indir, wp, bdtdir, pt), 'impacts.pdf'), outdir, suffix=f'__{wp}__{bdtdir}__{pt}')
                    mkdown_str += f'<object data="{plotname}" type="application/pdf" width="700px" height="500px"></object> \n'
                except Exception as e:
                    print(f'impact plot failed... {bdtdir}, {wp}, {pt}', '\n', e)
                    except_str = 'pT ({}, {}) vacant'.format(ptmin, ptmax if ptmax!=100000 else '+inf')
                    mkdown_str += f'<textarea name="a" style="width:700px;height:500px;">{except_str}</textarea> \n'
            mkdown_str += '\n'

        mkdown_str += '------------------\n'

    if args.show_unce_breakdown:
        title
        mkdown_str += '# cc-tagging SF uncetainty breakdown for syst. and stat. \n'
        for wp in scanned_wp_list:
            mkdown_str += f'## **{wpname[wp]}** WP: \nLeft to right: pT in ' + ', '.join(['({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + '\n\n'
            for pt, ptcut in zip(ptlist, ptcutlist):
                ptmin, ptmax = ptcut
                bdtdir = user_pt_bdt_map[pt]
                try:
                    plotname  = fetch_plot(os.path.join(get_dir(indir, wp, bdtdir, pt), 'unce_breakdown.png'), outdir, suffix=f'__{wp}__{bdtdir}__{pt}')
                    mkdown_str += f'<img src="{plotname}" title="{plottitle}" alt="{plottitle}" style="width:400px;"/> \n'
                except Exception as e:
                    print(f'unce breakdown plot failed... {bdtdir}, {wp}, {pt}', '\n', e)
                    except_str = 'pT ({}, {}) vacant'.format(ptmin, ptmax if ptmax!=100000 else '+inf')
                    mkdown_str += f'<textarea name="a" style="width:400px;height:400px;">{except_str}</textarea> \n'
            mkdown_str += '\n'
                    
    
    # make SF plots for varied sfBDT cut
    if args.bdt == 'auto' and (args.draw_sfbdt_vary or args.draw_sfbdt_vary_dryrun):
        mkdown_str += '# cc-tagging SF as a function of `sfBDT` cut position \n'
        mkdown_str += 'pT range | `sfBDT` variations (>x)\n -------- | --------- \n'
        for pt, ptcut in zip(ptlist, ptcutlist):
            ptmin, ptmax = ptcut
            mkdown_str += '({ptmin}, {ptmax}) | {bdtlist} \n'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf', bdtlist=', '.join([f'{int(b[3:])/1000.:.3f}' for b in user_pt_bdtvarylist_map[pt]]))
        mkdown_str += '\n\n'
        for wp in scanned_wp_list:
            mkdown_str += f'## **{wpname[wp]}** WP: \nLeft to right: pT in ' + ', '.join(['({}, {})'.format(ptmin, ptmax if ptmax!=100000 else '+inf') for ptmin, ptmax in ptcutlist]) + '\n\n'
            sf_list = [flv1, flv2, 'L']
            sf_draw = sf_list[:1] if not args.draw_sfbdt_vary_with_bl else sf_list
            for sf in sf_draw:
                for pt, ptcut in zip(ptlist, ptcutlist):
                    ptmin, ptmax = ptcut
                    plotname = f'sfbdtvary_{sf}__{wp}__{pt}.png'
                    plottitle = '({ptmin}, {ptmax})'.format(ptmin=ptmin, ptmax=ptmax if ptmax!=100000 else '+inf')
                    if not args.draw_sfbdt_vary_dryrun:
                        draw_sfbdtvary_plot(indir, wp, pt, ptcut, user_pt_bdtvarylist_map[pt], savepath=os.path.join(outdir, plotname), sf=sf)
                    mkdown_str += f'<img src="{plotname}" title="{plottitle}" alt="{plottitle}" style="width:400px;"/> \n'
                mkdown_str += '\n'

    # print (mkdown_str)
    with open(os.path.join(outdir, 'index.html'), 'w') as f:
        f.write(html_template.replace('$TITLE', outdir.split('_allWP_')[-1]).replace('$TEXT', mkdown_str))

    

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
  <button onclick="replaceYear(2016)">2016</button>
  <button onclick="replaceYear(2017)">2017</button>
  <button onclick="replaceYear(2018)">2018</button>
  <script>
  function replaceYear(y) {
    location.href=location.href.replace(new RegExp('201[678]'), y);
  }
  </script>

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

import yaml
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, args.cfg)) as f:
    config = yaml.safe_load(f)

if config['type'].lower() == 'cc':
    flv1, flv2 = 'C', 'B'
elif config['type'].lower() == 'bb':
    flv1, flv2 = 'B', 'C'
else:
    raise RuntimeError(f'Tagger type in {args.cfg} must be cc or bb.')

wp_list = list(config['tagger']['working_points']['range'].keys())
wp0 = wp_list[0]

## Global vars
lumi = {2016: 35.92, 2017: 41.53, 2018: 59.74}
wpname_dict = {'HP':'High purity', 'TP':'High purity', 'MP':'Medium purity', 'LP':'Low purity'}
wpname = {wp:(wpname_dict[wp] if wp in wpname_dict else wp) for wp in wp_list}
unce_list = ['pu','fracBB','fracCC','fracLight','psWeightIsr','psWeightFsr','sfBDTRwgt']
if args.ext_unce is not None:
    for ext_unce in args.ext_unce.split(','):
        if not ext_unce.startswith('~'):
            unce_list.append(ext_unce)
        else:
            unce_list.remove(ext_unce[1:])

dic_ptlist = {}
dic_user_pt_bdtvarylist_map = {}

# Note that same fit points for all WPs e.g. {LP, MP, HP} will be combined in one page

from cmssw.utils import find_valid_runlist

for inputdir in find_valid_runlist(args.dir, bdt_mode=args.bdt):
    # Read the basedir, name of pt, bdt from the inputdir
    import re
    basedir = re.findall('[0-9]{8}_[\w-]+', inputdir)
    pt = re.findall('/(pt[0-9]+to[0-9]+)/', inputdir)
    bdt = re.findall('/(bdt[0-9]+)', inputdir)
#     print(inputdir, basedir, pt, bdt)
    assert len(basedir)==1 and len(pt)==1 and len(bdt)==1
    basedir, pt, bdt = basedir[0], pt[0], bdt[0]

    # Generate two dics: dic_ptlist & dic_user_pt_bdtvarylist_map
    if basedir not in dic_ptlist:
        dic_ptlist[basedir] = []
    if pt not in dic_ptlist[basedir]:
        dic_ptlist[basedir].append(pt)

    if basedir not in dic_user_pt_bdtvarylist_map:
        dic_user_pt_bdtvarylist_map[basedir] = {}
    if pt not in dic_user_pt_bdtvarylist_map[basedir].keys():
        dic_user_pt_bdtvarylist_map[basedir][pt] = []
    dic_user_pt_bdtvarylist_map[basedir][pt].append(bdt)

# Find out the dic_user_pt_bdt_map to process.
#  - if set `--bdt auto`; then use the central sfBDT value to generate the webpage
#  - if set e.g. `--bdt 900`; then use the user specified value
import copy
dic_user_pt_bdt_map = copy.deepcopy(dic_user_pt_bdtvarylist_map)
for basedir in dic_user_pt_bdtvarylist_map.keys():
    for pt in dic_user_pt_bdtvarylist_map[basedir].keys():
        dic_user_pt_bdtvarylist_map[basedir][pt] = sorted(dic_user_pt_bdtvarylist_map[basedir][pt])
        n_bdt = len(dic_user_pt_bdtvarylist_map[basedir][pt])
        if args.bdt != 'auto':
            assert 'bdt'+args.bdt in dic_user_pt_bdtvarylist_map[basedir][pt]
            dic_user_pt_bdt_map[basedir][pt] = 'bdt'+args.bdt
        else:
            dic_user_pt_bdt_map[basedir][pt] = dic_user_pt_bdtvarylist_map[basedir][pt][int(n_bdt/2)]

for indir in glob.glob(args.dir):
    if f'_{wp0}_' in indir:
        basedir = re.findall('[0-9]{8}_[\w-]+', indir)[0]
        make_website(
            indir,
            basedir,
            outdir=make_new_dir(indir_name=basedir, outbasedir=args.outweb),
            ptlist=sorted(dic_ptlist[basedir], key=lambda pt: int(pt.split('to')[0].split('pt')[1])), 
            user_pt_bdt_map=dic_user_pt_bdt_map[basedir],
            user_pt_bdtvarylist_map=dic_user_pt_bdtvarylist_map[basedir],
            scanned_wp_list=wp_list,
        )