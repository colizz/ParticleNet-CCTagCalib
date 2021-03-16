#!/usr/bin/env python3

import os
import glob
import argparse
parser = argparse.ArgumentParser('create all fit routine')
parser.add_argument('--dir', default=None, help='Path to the base directory of ROOT output.')
parser.add_argument('--ext-unce', default=None, help='Argument pass to write_cards_sf: set extra uncertainty term to run. e.g. --ext-unce NewTerm1,NewTerm2')
parser.add_argument('--run-impact', action='store_true', help='Argument pass to write_cards_sf: to run impact plots.')
parser.add_argument('--run-unce-breakdown', action='store_true', help='Argument pass to write_cards_sf: to run uncertainty breakdown')
parser.add_argument('--bdt', default='900', help='The BDT folder to run. Set e.g. `--bdt 840,860,880,900,920,940` or `--bdt auto`.')
parser.add_argument('-t', '--threads', type=int, default=8, help='Concurrent threads to run separate fits.')
args = parser.parse_args()

if not args.dir:
    raise RuntimeError('--dir is not provided!')

n_fits = 0
str_runfit = ''
ext_cmd = ''
if args.ext_unce is not None:
    ext_cmd += f'--ext-unce {args.ext_unce} '
if args.run_impact:
    ext_cmd += f'--run-impact '
if args.run_unce_breakdown:
    ext_cmd += f'--run-unce-breakdown '

from utils import find_valid_runlist

for inputdir in find_valid_runlist(args.dir, bdt_mode=args.bdt):
    if not os.path.exists(inputdir):
        raise RuntimeError('Input directory does not exists: ', inputdir)
    n_fits += 1
    cmd_runfit = f'python write_cards_sf.py {inputdir} {ext_cmd} &'

    str_runfit += cmd_runfit+'\n'
    
    if n_fits % args.threads == 0:
        str_runfit += 'wait\n'
str_runfit += 'wait\n'

print (str_runfit)  
with open('bg_runfit.sh', 'w') as fw:
    fw.write(str_runfit)

print('File bg_runfit.sh is created.')