#!/usr/bin/env python3

import os
import glob
import argparse
parser = argparse.ArgumentParser('create all fit routine')
parser.add_argument('--dir', default=None, help='Path to the base directory of ROOT output.')
parser.add_argument('--full', action='store_true', help='Argument pass to write_cards_sf: to run the full fit routine including the implact.')
parser.add_argument('--bdt', default='900', help='The BDT folder to run. Only in the routine for BDT varying validation we set --bdt 840,860,880,900,920,940')
args = parser.parse_args()

if not args.dir:
    raise RuntimeError('--dir is not provided!')

abspath = os.path.abspath(os.getcwd())
str_runfit = ''
for sam in glob.glob(args.dir):
    for bdtval in args.bdt.split(','):
        bdt = 'bdt'+bdtval
        for pt in os.listdir(os.path.join(abspath, sam, 'Cards', bdt)):
            inputdir = os.path.join(abspath, sam, 'Cards', bdt, pt)
            cmd_runfit = '. write_cards_sf_wrapper.sh '+inputdir+(' full &' if args.full else ' &')
            str_runfit += cmd_runfit+'\n'
            print (cmd_runfit)
    
with open('bg_runfit.sh', 'w') as fw:
    fw.write(str_runfit)
print('File bg_runfit.sh is created.')