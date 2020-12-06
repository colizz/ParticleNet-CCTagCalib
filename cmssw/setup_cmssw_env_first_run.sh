#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
cmsrel CMSSW_10_2_18
cd CMSSW_10_2_18/src
cmsenv

## Install HiggsAnalysis
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cp ../../data/TagAndProbeExtendedV2.py HiggsAnalysis/CombinedLimit/python/  # copy the model we will use in fit
cd HiggsAnalysis/CombinedLimit
source env_standalone.sh
cd ../..

## Install CombineHarvester
git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
scram b -j8

cd ../..