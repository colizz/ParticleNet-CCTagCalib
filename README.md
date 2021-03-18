# ParticleNet-CCTagCalib

This mini repository aims to derive ParticleNet AK15 cc-tagger scale factors (SF), based on the g->cc proxy jet method.
The introduction of the method can be found in [these slides (final version: Mar.8)](https://indico.cern.ch/event/1014620/contributions/4265127/attachments/2203682/3728092/20.03.08_BTV_Update%20on%20ParticleNet%20SF%20method.pdf). A detailed documentation is provided in [AN-21-005](https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005). All derived SFs are summarized in [this link](https://coli.web.cern.ch/coli/repo/ParticleNet-CCTagCalib/sf_summary).

The main idea is to use similar characteristics between the resonance double charm (cc) jet and the g->cc splitting jet, the latter confined in a specific phase-space. 
By deriving the SFs for the latter, we can transfer the SFs and use them in the H->cc jet. 
With the universality of the method, it is also possible to derive the SFs for other deep boosted-jet taggers (e.g. DeepAK*X* tagger for AK8 or AK15), and/or for the bb-tagger (by using similarities with g->bb jets).

Below we will present how to run the code using a test dataset. 
Following all the scripts documented below and in the notebook, one can reproduce the SFs in the test dataset.

## Startup

To startup, please clone the repository

```shell
git clone https://github.com/colizz/ParticleNet-CCTagCalib.git
cd ParticleNet-CCTagCalib/
```

and download the test dataset. The dataset is stored on CERNBox and set with CMS permission. You can do a rsync or scp from lxplus.

```shell
rsync -a --progress <user>@lxplus.cern.ch:/eos/user/c/coli/cms-repo/ParticleNet-CCTagCalib/samples .
```

The framework to produce the dataset is provided in the appendix.

## Environment

The code requires python3, with the dependency of packages: `ROOT`, `uproot`, `boost_histogram`, `pandas`, `seaborn`, `xgboost`. 
It also requires `jupyter` to run the notebook. We recommend to use the `Miniconda`.

It is doable to install the `Miniconda`, then easily restore the conda environment<sup>*</sup> using the yaml config provided (do it only at the first time):

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda  # for test: put the miniconda folder here
source miniconda/bin/activate
## Create new env "cctag" defined in yml. This may take for a while
conda env create -f conda_env.yml
```

The `cctag` conda environment only needs to be created once. After that, one can simply activate the `cctag` env by:

```shell
conda activate cctag
```

**Alternatively**, one can do the following to update the existing `cctag` env using the latest yaml file.

```
conda env update --name cctag --file conda_env.yml
```

> (*) **Note**: we use the specific `xgboost` version **0.72.1** instead of the latest (1.2) in order to match with the pre-intalled version in `CMMSSW_10_6_18`, as we may use `Miniconda` env for BDT training and the latter for prediction. 
> (Need to handle with caution, since no obvious error message is reported even if the versions are not compatible.)

Apart from the `Miniconda`, we also use the `CMMSSW_10_6_18` environment to run Higgs combine tool for the fit process.

## Training and application of BDT

Prior to the fit, we introduce a BDT variable (named sfBDT) which helps to discriminant the gluon contaminated jet versus the clean g->cc jet in the QCD cc-jet candidates. 
The sfBDT is then used to define the g->cc phase space to derive the SFs. Please see the slides in the begining for more details.

**If you are only interested in the fit, you can simply skip this section**, as the dedicated samples we provided in the next section already include the sfBDT variable for jets.

The following code shows how we train the sfBDT from the QCD multijet samples:

```shell
python -u sfbdt_xgb.py --model-dir sfbdt_results --train
```

The test training has `kfold=2` iterations that use the half events for training and predicts on the other half, iteratively. 
In reality, we use `kfold=10` (i.e., to train on 9/10 and to predict on the rest 1/10, iteratively). 
The output BDT models are stored in `sfbdt_results/`.

For the application step, we run:

```shell
python -u sfbdt_xgb.py --model-dir sfbdt_results --predict --jet-idx 1 --bdt-varname fj_1_sfBDT_test -i samples/trees_sf/20201028_nohtwbdt_v2_ak15_qcd_2016 -o samples/trees_sf/20201028_nohtwbdt_v2_ak15_qcd_2016__bdt_fj1  ## predict score for the leading jet
python -u sfbdt_xgb.py --model-dir sfbdt_results --predict --jet-idx 2 --bdt-varname fj_2_sfBDT_test -i samples/trees_sf/20201028_nohtwbdt_v2_ak15_qcd_2016 -o samples/trees_sf/20201028_nohtwbdt_v2_ak15_qcd_2016__bdt_fj2  ## predict score for the subl. jet
```

## SF derivation

### Setup `config.yml`

In `config.yml` we need to provide necessary settings for SF derivation. In our current setup, we specify the sample directory and the year condition to be 2018. The information of tagger (the variable name, type, and the working point setup) and the pT range of fit is also provided in the file.

### Derive the templates for fit

We first derive the histogram templates in the ROOT-format used as the input to Higgs combine tool. To do that, please start a Jupyter service (already installed properly in conda) and open the notebook `ak15_sf_main_ak.ipynb`. **All details are documented in this notebook.**

(Note: alternatively, one can use `ak15_sf_main_pd.ipynb` which has the same goal but is implemented by pandas DataFrame. This notebook may consume more RAM but can runs faster.)

After complete the first section ("make templates for fit") of the notebook, we can find our output files stored in `results/`.

### Implement the fit

 `CMMSSW_10_6_18` is required to run the Higgs combine tool. Please note that you can use the CMS singularity container by doing `cmssw-cc7` if the machine is not running a cc7 system.

**In a cc7 environment**, by sourcing the following script, we can load the cms-sw environment and install the package `HiggsAnalysis` and `CombineHarvester` for the first run:

```shell
cd cmssw/
source setup_cmssw_env_first_run.sh
```

Then, we create the launch script for all the fit points, and start to run each point as an individual background process.
The fit is implemented by the Higgs combine tool. We run every individual fit points with the sfBDT cut at all variation values.

```shell
./create_all_fit_routine.py --dir '../results/20210315_SF201?_AK15_qcd_ak_pnV02_?P_*' --bdt auto -t 10
source bg_runfit.sh
```

When all jobs are finished, we may optionally run the full suite of fit (including the impact plot and the uncertainty breakdown) for specific fit points where sfBDT cut at the central points. Or we can go directly to the next step.

```shell
./create_all_fit_routine.py --dir '../results/20210315_SF201?_AK15_qcd_ak_pnV02_?P_*' --bdt central -t 10 --run-impact --run-unce-breakdown
source bg_runfit.sh 
```

### Organize fit results

When all jobs are finished, we **return to the conda environment**:

````shell
cd ../ && conda activate cctag
````

and produce all necessary plots<sup>†</sup> using the script below. The idea is to read the pre-fit and post-fit information from `fitDiagnostics.root`, which is generated by the Higgs combine tool. Note that we only need to make plots for the fit with sfBDT cut at the central value.

```shell
./make_plots.py --dir 'results/20210315_SF201?_AK15_qcd_ak_pnV02_?P_*' --bdt central -t 10
```

Finally, we can summarize all these results (SFs, histograms, impact plots...) on a webpage. The argument `--draw-sfbdt-vary` will create additional plots for the fitted SFs varying as a function of the sfBDT cut value.

```shell
./make_html.py --dir 'results/20210315_SF201?_AK15_qcd_ak_pnV02_?P_*' --bdt auto --outweb web/testdir --draw-sfbdt-vary --show-unce-breakdown
```

It extracts the SFs from the log file, copies all the plots to the output folder and writes an HTML to organize them in a neat way. 
After that we can open the file `web/testdir/<sample name>/index.html` in the web browser. 
It should be very much like this [example webpage](https://coli.web.cern.ch/coli/repo/ParticleNet-CCTagCalib/exampleweb/bdt900). Now the work is done -- please enjoy ;)

> (†) **Note**: The code to produce the plots in section "pre/post-fit template" is currently not valid in this repo due to other dependency issues, but since all information is in `fitDiagnostics.root`, they are technically doable to reproduce.

### Run additional fit with argument `fitVarRwgt`

In V2 of the framework, we introduce an additional uncertainty term obtained from an alternative fit that applies a reweighting on the fit variable in the "pass+fail" inclusive MC. To obtain this uncertainty, we need to re-run the fit, collect the fitted SFs, calculate the uncertainty, then form the webpage. As the templates for this uncertainty are already produced in the subfolder `fitVarRwgt(Up|Down)`, we simply re-run the fit with this extra uncertainty source included via the argument ` --ext-unce fitVarRwgt`

```shell
## Make a clean copy of the produced templates
mkdir results/fitvarrwgt
\cp -ar results/20210315_SF201?_AK15_qcd_ak_pnV02_?P_* results/fitvarrwgt/
\rm -f results/fitvarrwgt/20210315_SF201?_AK15_qcd_ak_pnV02_?P_*/*/*/*/*.*
```

```shell
## Load the cmssw env and run fit
conda deactivate
cd cmssw/CMSSW_10_2_18/src/ && cmsenv && cd ../..
./create_all_fit_routine.py --dir '../results/fitvarrwgt/20210315_SF201?_AK15_qcd_ak_pnV02_?P_*' --bdt central -t 10 --ext-unce fitVarRwgt
source bg_runfit.sh
```

Then make the webpage again. This time we append argument `--show-fitvarrwgt-unce` to show the new uncertainty source in the webpage. (`--draw-sfbdt-vary-dryrun`: no need to reproduce the sfBDT varying plots again but keep this section on HTML.)

```shell
## Return to the conda env, then make the webpage again
cd ..
conda activate cctag
./make_html.py --dir 'results/20210315_SF201?_AK15_qcd_ak_pnV02_?P_*' --bdt auto --outweb web/testdir --draw-sfbdt-vary-dryrun --show-unce-breakdown --show-fitvarrwgt-unce
```

## Appendix

### Produce the input datasets

The input ntuples are produced using [NanoHRT-tools](https://github.com/hqucms/NanoHRT-tools), which is based on the CMS utilitiy [nanoAOD-tools](https://github.com/cms-nanoAOD/nanoAOD-tools).

The framework takes NanoAOD as inputs and writes out flat ntuples. In our workflow, the input samples are QCD multijet events (which is dominant) as well as heavy resonance contribution from the ttbar, single top and V+jets, see the sample config file e.g. [here](https://github.com/hqucms/NanoHRT-tools/blob/master/run/custom_samples/qcd_2018_MC.yaml). Currently we use the branch [`dev/unify-producer`](https://github.com/hqucms/NanoHRT-tools/tree/dev/unify-producer).

Please follow the introduction in the above link. Note that we shall use `CMSSW_11_1_0_pre5_PY3` to support `ONNXRuntime` in order to use the ML infrastructure to re-run the trigger value, as mentioned in README. If the tagger score cannot be obtained from the current NanoAOD variable, we may need to consider produce an alternative set of NanoAODs to equip with the score.

The script to produce the year 2017 of our test ntuples (on lxplus):

```shell
python runHeavyFlavTrees.py -i /eos/cms/store/cmst3/group/vhcc/nanoTuples/v2_30Apr2020/2017/mc/   -o <output-path>/20210102_pnV02 --sample-dir custom_samples --jet-type ak15 --channel qcd --year 2017
python runHeavyFlavTrees.py -i /eos/cms/store/cmst3/group/vhcc/nanoTuples/v2_30Apr2020/2017/data/ -o <output-path>/20210102_pnV02 --sample-dir custom_samples --jet-type ak15 --channel qcd --year 2017 --run-data
```

