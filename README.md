# ParticleNet-CCTagCalib

> The repo will be renamed as it has been upgraded to a general tool to calibrate any boosted-jet tagger for bb/cc type.

This mini repository aims to derive ParticleNet AK15 cc-tagger scale factors (SF), based on the g->cc proxy jet method.
The introduction of the method can be found in [these slides (final version: Mar.8)](https://indico.cern.ch/event/1014620/contributions/4265127/attachments/2203682/3728092/20.03.08_BTV_Update%20on%20ParticleNet%20SF%20method.pdf). Detailed documentation is provided in [AN-21-005](https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005). All derived SFs are summarized in [this link](https://coli.web.cern.ch/coli/repo/ParticleNet-CCTagCalib/sf_summary).

The main idea is to use similar characteristics between the resonance double charm (cc) jet and the g->cc splitting jet, the latter confined in a specific phase-space. 
By deriving the SFs for the latter, we can transfer the SFs and use them in the H->cc jet. 
With the universality of the method, it is also possible to derive the SFs for other deep boosted-jet taggers (e.g. DeepAK*X* tagger for AK8 or AK15), and/or for the bb-tagger (by using similarities with g->bb jets).

Below we will present how to run the code using a test dataset, to calibrate the ParticleNet Xbb tagger for AK8 jets.
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
It also requires `jupyter` to run the notebook. We recommend using the `Miniconda`.

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

> (*) **Note**: we use the specific `xgboost` version **0.72.1** instead of the latest (1.2) in order to match with the pre-installed version in `CMMSSW_10_6_18`, as we may use `Miniconda` env for BDT training and the latter for prediction. 
> (Need to handle with caution, since no obvious error message is reported even if the versions are not compatible.)

Apart from the `Miniconda`, we also use the `CMMSSW_10_6_18` environment to run Higgs combine tool for the fit process.

## Training and application of BDT

> To be updated

Prior to the fit, we introduce a BDT variable (named sfBDT) which helps to discriminant the gluon contaminated jet versus the clean g->cc jet in the QCD cc-jet candidates. 
The sfBDT is then used to define the g->cc phase space to derive the SFs. Please see the slides for more details.

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

### Setup config YML

In `cards/` we have different YML files as the configuration in different calibration scenarios. Specified in YML are the routine name, the bb/cc type for calibration, the input sample info, the tagger info, the pT ranges, as well as the info for main analysis trees used as a proxy. Detailed instruction is given in the example cards. For the standard ParticleNet Xbb calibration for AK8 jets, please use `cards/config_bb_std.yml`.

### Preprocess

In the step of preprocessing, the input samples are read into the framework to produce new variables used for making the templates and to extract necessary reweight factors. Please start a Jupyter service (already installed properly in conda) and run through the notebook `preprocessing.ipynb`. The intermediate files will be stored in the `prep` folder.

### Derive the templates for fit

We then derive the histogram templates in the ROOT-format used as the input to Higgs combine tool. Two notebooks, `make_template_pd.ipynb` and `make_template_ak.ipynb`, are designed with the same goal, using `pandas` data frame and awkward-array respectively for event processing. Each has its advantage in time or RAM saving, so you can use either one of them to derive the templates. All details are documented in this notebook.

After making the templates for the main routine (see notebook), we can quickly continue to derive the SFs. Other routines are for validation purposes. The templates are stored in `results/`.

### Two environments

The below steps are run in two environments that we need to distinguish. **All codes under the folder `cmssw` are run in the `CMSSW_10_6_18` env with the Higgs combine tool set up. The other code outside `cmssw` are running in the anaconda `cctag` env.**

Given this rule, we will not mention the environment used below but just to make sure the code is running in the correct environment.

### Implement the fit

 `CMMSSW_10_6_18` is required to run the fit using the Higgs combine tool. In a cc7 environment, by sourcing the following script, we can load the cms-sw environment and install the package `HiggsAnalysis` and `CombineHarvester` for the first run:

```shell
cd cmssw/
source setup_cmssw_env_first_run.sh
```

Then, we create the launch script for all the fit points, and start to run each point as an individual background process.
The fit is implemented by the Higgs combine tool. We run every individual fit points with the sfBDT cut at all variation values.

```shell
./create_all_fit_routine.py --dir '../results/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt auto -t 10
source bg_runfit.sh
```

Here, `--bdt auto` means we run over all BDT cut points as an individual fit, as their results are all required. `-t 10` specifies the threads for the concurrent run. 

When all jobs are finished, we may run the full suite of fit (including the impact plot and the uncertainty breakdown) for specific fit points where sfBDT cut at the central points.

```shell
./create_all_fit_routine.py --dir '../results/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt central -t 10 --run-impact --run-unce-breakdown
source bg_runfit.sh
```

### (*) Optional fit for the modified sfBDT cut scheme

If the variable `bdt_mod_factor` is not set to `None` during the preprocessing step 3-2, we need to do alternative fit by switching to the modified sfBDT cut scheme as the results are also needed. The template making notebook should already produced corresponding templates under `results/bdtmod/`. Please run

```shell
./create_all_fit_routine.py --dir '../results/bdtmod/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt auto -t 10
source bg_runfit.sh
```

### Organize fit results

When all jobs are finished, we open a clean shell and enter the anaconda `cctag` environment.

We then produce all necessary plots<sup>†</sup> using the script below. The idea is to read the pre-fit and post-fit information from `fitDiagnostics.root`, which is generated by the Higgs combine tool. Note that we only need to make plots for the fit with sfBDT cut at the central value.

```shell
./make_plots.py --dir 'results/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt central -t 10
```

Finally, we can summarize all these results (SFs, histograms, impact plots...) on a webpage. The argument `--draw-sfbdt-vary` will create additional plots for the fitted SFs varying as a function of the sfBDT cut value.

```shell
./make_html.py --dir 'results/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt auto --outweb web/Xbb --draw-sfbdt-vary --show-unce-breakdown --combine-bdtmod
```

It extracts the SFs from the log file, copies all the plots to the output folder and writes an HTML to organize them in a neat way. 
After that we can open the file `web/Xbb/<sample name>` in the web browser. 
It should be very much like this [example webpage](https://coli.web.cern.ch/coli/repo/ParticleNet-CCTagCalib/exampleweb/bdt900).

> (†) **Note**: The code to produce the plots in section "pre/post-fit template" is currently not valid in this repo due to other dependency issues, but since all information is in `fitDiagnostics.root`, they are technically doable to reproduce.

### (*) Include modified sfBDT scheme results in the web

The fit results from the modified sfBDT scheme can be collected and shown on the webpage.

```shell
./make_plots.py --dir 'results/bdtmod/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt central -t 10
./make_html.py --dir 'results/bdtmod/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt auto --outweb web/Xbb_bdtmod --draw-sfbdt-vary
```

Besides, following the strategy that we combine the results from both schemes to determine the "max d" uncertainty source, we re-make the original web by adding `--combine-bdtmod`.

```shell
./make_html.py --dir 'results/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt auto --outweb web/Xbb --draw-sfbdt-vary-dryrun --show-unce-breakdown --combine-bdtmod
```

The webpage is updated under `web/Xbb/<sample name>`.

### Run additional fit with argument `fitVarRwgt`

In V2 of the framework, we introduce an additional uncertainty term obtained from an alternative fit that applies a reweighting on the fit variable in the "pass+fail" inclusive MC. To obtain this uncertainty, we need to re-run the fit, collect the fitted SFs, calculate the uncertainty, then form the webpage. As the templates for this uncertainty are already produced in the subfolder `fitVarRwgt(Up|Down)`, we simply re-run the fit with this extra uncertainty source included via the argument ` --ext-unce fitVarRwgt` 

First, make a clean copy of the produced templates under `results/fitvarrwgt`.

```shell
mkdir results/fitvarrwgt
\cp -ar results/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2 results/fitvarrwgt/
\rm -f results/fitvarrwgt/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2/Cards/*/*/*.*
```

Then run the fit on these templates in the `CMSSW_10_2_18` env, with the extra uncertainty source `--ext-unce fitVarRwgt`.

```shell
./create_all_fit_routine.py --dir '../results/fitvarrwgt/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt central -t 10 --ext-unce fitVarRwgt
source bg_runfit.sh
```

Then make the webpage again in the conda `cctag` env. This time we append argument `--show-fitvarrwgt-unce` to show the new uncertainty source in the webpage. The main webpage under `web/Xbb` is updated. The results produced from the `fitVarRwgt` scheme is also summarised in `web/Xbb_fitvarrwgt`.

```shell
./make_html.py --dir 'results/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt auto --outweb web/Xbb --draw-sfbdt-vary-dryrun --show-unce-breakdown --combine-bdtmod --show-fitvarrwgt-unce
./make_html.py --dir 'results/fitvarrwgt/20210324_bb_SF201?_pnV02_?P_msv12_dxysig_log_var22binsv2' --cfg cards/config_bb_std.yml --bdt auto --outweb web/Xbb_fitvarrwgt --show-fit-number-only
```

## Post histogram making

The notebook `post_hist.ipynb` provides code to make different types of histogram, using the backup files created in the preprocessing step. See more details in the notebook.

## Appendix

### Produce the input datasets

The input ntuples are produced using [NanoHRT-tools](https://github.com/hqucms/NanoHRT-tools), which is based on the CMS utilitiy [nanoAOD-tools](https://github.com/cms-nanoAOD/nanoAOD-tools).

The framework takes NanoAOD as inputs and writes out flat ntuples. In our workflow, the input samples are QCD multijet events (which is dominant) as well as heavy resonance contribution from the ttbar, single top and V+jets, see the sample config file e.g. [here](https://github.com/hqucms/NanoHRT-tools/blob/master/run/custom_samples/qcd_2018_MC.yaml). Currently we use the branch [`dev/unify-producer`](https://github.com/hqucms/NanoHRT-tools/tree/dev/unify-producer).

Please follow the introduction in the above link. The script to produce the ntuples (i.e. the test datasets used above) to calibrate the ParticleNet Xbb AK8 tagger is shown below.

For EOY datasets, we use the custom NanoAOD to derive the ntuples. For year condition 2016, do the following on lxplus (for year 2017, replace all 2016 to 2017).

```shell
python runHeavyFlavTrees.py -i /eos/cms/store/cmst3/group/vhcc/nanoTuples/v2_30Apr2020/2016/mc/   -o <output-path>/particlenet_ak8_20210113 --sample-dir custom_samples --jet-type ak8 --channel qcd --year 2016
python runHeavyFlavTrees.py -i /eos/cms/store/cmst3/group/vhcc/nanoTuples/v2_30Apr2020/2016/data/ -o <output-path>/particlenet_ak8_20210113 --sample-dir custom_samples --jet-type ak8 --channel qcd --year 2016 --run-data
```

For EOY datasets in year condition 2018, do the following on FNAL cluster.

```shell
python runHeavyFlavTrees.py -i /eos/uscms/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/mc/   -o <output-path>/particlenet_ak8_20210113 --sample-dir custom_samples --jet-type ak8 --channel qcd --year 2018
python runHeavyFlavTrees.py -i /eos/uscms/store/group/lpcjme/nanoTuples/v2_30Apr2020/2018/data/ -o <output-path>/particlenet_ak8_20210113 --sample-dir custom_samples --jet-type ak8 --channel qcd --year 2018 --run-data
```

For UL datasets, the framework is under development.

Note: after the condor jobs complete, please remember to run the same command appended with `--post`. This procedure post-process the output ROOT file (adding `xsecWeight`) then combine them into one per sample. See more in [NanoHRT-tools](https://github.com/hqucms/NanoHRT-tools) README.