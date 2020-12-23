# ParticleNet-CCTagCalib

This mini repository aims to derive ParticleNet AK15 cc-tagger SFs, based on the g->cc proxy jets method. 
The introduction of the method can be found in [these slides (final version: Dec.7)](https://indico.cern.ch/event/980437/contributions/4134498/attachments/2158018/3640299/20.12.07_BTV_ParticleNet%20cc-tagger%20calibration%20for%20AK15%20jets%20using%20the%20g-_cc%20method.pdf). All derived SFs are summarized in [this link](https://coli.web.cern.ch/coli/repo/ParticleNet-CCTagCalib/sf_summary).

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

## Environment

The code requires python3, with the dependency of packages: `ROOT`, `uproot`, `boost_histogram`, `pandas`, `seaborn`, `xgboost`. 
It also requires `jupyter` to run the notebook. We recommend to use the `Miniconda`.

 It is doable to install the `Miniconda`, then easily restore the conda environment<sup>*</sup> using the yaml config provided:

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda  # for test: put the miniconda folder here
source miniconda/bin/activate
## Create new env "cctag" defined in yaml. This may take for a while
conda env create -f conda_env.yaml
```

The `cctag` conda environment only needs to be created once. After that, one can simply activate the `cctag` env by:

```shell
conda activate cctag
```

> (*) **Note**: we use the specific `xgboost` version **0.72.1** instead of the latest (1.2) in order to match with the pre-intalled version in `CMMSSW_10_6_18`, as we may use `Miniconda` env for BDT training and the latter for prediction. 
> (Need to handle with caution, since no obvious error message is reported even if the versions are not compatible.)

Apart from the `Miniconda`, we also use the `CMMSSW_10_6_18` environment to run Higgs Combine tool for the fit process.

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
python -u sfbdt_xgb.py --model-dir sfbdt_results --predict --jet-idx 1 --bdt-varname fj_1_sfBDT_test  -i samples/trees_sf/20201028_nohtwbdt_v2_ak15_qcd_2016 -o samples/trees_sf/20201028_nohtwbdt_v2_ak15_qcd_2016__bdt_fj1  ## predict score for the leading jet
python -u sfbdt_xgb.py --model-dir sfbdt_results --predict --jet-idx 2 --bdt-varname fj_2_sfBDT_test -i samples/trees_sf/20201028_nohtwbdt_v2_ak15_qcd_2016 -o samples/trees_sf/20201028_nohtwbdt_v2_ak15_qcd_2016__bdt_fj2  ## predict score for the subl. jet
```

## SF derivation

### Derive the templates for fit

We first derive the histogram templates in the ROOT-format used as the input to Higgs Combine tool. To do that, please start a Jupyter service (already installed properly in conda) and open the notebook `ak15_sf_main_ak.ipynb`. **All details are documented in this notebook.**

After complete the first section ("make templates for fit") of the notebook, we can find our output files stored in `results/`.

### Implement the fit

 `CMMSSW_10_6_18` is required to run the Higgs Combine tool. Please note that you can use the CMS singularity container by doing `cmssw-cc7` if the machine is not running a cc7 system.

**In a cc7 environment**, by sourcing the following script, we can load the cms-sw environment and install the package `HiggsAnalysis` and `CombineHarvester` for the first run:

```shell
cd cmssw/
source setup_cmssw_env_first_run.sh
```

Then, we create the launch script for all the fit points, and start to run each point as an individial background process.
The fit is implemented by the Higgs Combine.

```shell
./create_all_fit_routine.py --dir '../results/*' --full  ## create the launch script
source bg_runfit.sh  ## run the script
```

As all processes run in the background, one can use `jobs` to check the status in the runtime.

When all jobs are finished, we can **return to the conda environment**:

````shell
cd ../ && conda activate cctag
````

and produce all necessary plots<sup>†</sup> using the script below. The idea is to read the pre-fit and post-fit information from `fitDiagnostics.root`, which is generated by the Higgs Combine.

```shell
./make_plots.py --dir 'results/*'
```

Finally, we can summarize all these results (SFs, histograms, impact plots...) on a webpage.

```shell
./make_html.py --dir 'results/*' --outweb web/testdir
```

It extracts the SFs from the log file, copies all the plots to the output folder and writes an HTML to organize them in a neat way. 
After that we can open the file `web/testdir/<sample name>/bdt900` in the web browser. 
It should be very much like this [example webpage](https://coli.web.cern.ch/coli/repo/ParticleNet-CCTagCalib/exampleweb/bdt900). Now the work is done -- please enjoy ;)

> (†) **Note**: The code to produce the plots in section "pre/post-fit template" is currently not valid in this repo due to other dependency issues, but since all information is in `fitDiagnostics.root`, they are technically doable to reproduce.