from __future__ import print_function

import os
import pdb
import shutil

import xgboost as xgb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import uproot
import pandas as pd

import argparse
parser = argparse.ArgumentParser('xgboost training and application')
parser.add_argument('--train', action='store_true', help='Do training.')
parser.add_argument('--tune', action='store_true', help='Do hyperparam tuning.')
parser.add_argument('--predict', action='store_true', help='Do testing.')
parser.add_argument('--gpu', default='', help='gpu number')
parser.add_argument('--model-dir', help='Where to save/load the xgb model.')
parser.add_argument('--bdt-varname', help='BDT variable name to store.')
parser.add_argument('--jet-idx', default=2, choices=['1', '2'], help='Jet index used for training/predicting.')
parser.add_argument('-i', '--inputdir', help='Input dir')
parser.add_argument('-o', '--outputdir', help='Output dir')
args = parser.parse_args()

if not args.model_dir:
    raise RuntimeError('--model-dir is not provided!')
if args.predict and not args.bdt_varname:
    raise RuntimeError('--bdt-varname is not provided!')
x = args.jet_idx ## jet index noted as "x"

# ---------------------------------------------
# preprocessing
basedirs = [
    'samples/trees_sf/20200622_addGenBC_allTypeSV_ak15_qcd_2016/mc', # sample dedicated for training
]
flist_train = [
    ('qcd', 'qcd-mg_tree.root'),
]
flist_pred = [
    'mc/ggH-cc_tree.root',
    'mc/qcd-mg_tree.root',
    'mc/qcd-herwig_tree.root',
    'mc/v-qq_tree.root',
    'data/jetht_tree.root',
]

basesel = f'passHTTrig & ht>1000 & fj_{x}_dr_Z>2.5 & fj_{x}_pt>200 & fj_{x}_sdmass>50 & fj_{x}_sdmass<200 & passmetfilters & fj_{x}_nbhadrons==0 & fj_{x}_sj1_nchadrons>=1 & fj_{x}_sj2_nchadrons>=1'

# ---------------------------------------------
k_folds = 2

model_name = 'xgb_train.model'

sigexpr = f'fj_{x}_gpart_ptratio < 0.15'; bkgexpr = f'fj_{x}_gpart_ptratio > 0.85'
wgtvar = 'train_weight'
# wgtexpr = 'genWeight*xsecWeight*puWeight'
wgtexpr = '1'
label_var = 'flag_signal'

######
train_vars = [
    f'fj_{x}_tau21',
    f'fj_{x}_sj1_rawmass',
    f'fj_{x}_sj2_rawmass',
    f'fj_{x}_ntracks_sv12',
    f'fj_{x}_sj1_sv1_pt',
    f'fj_{x}_sj2_sv1_pt',
]
obs_vars = [
    'event',
    'passmetfilters',
    'passHTTrig',
    'ht',
    f'fj_{x}_dr_Z',
    f'fj_{x}_pt',
    f'fj_{x}_sdmass',
    f'fj_{x}_nbhadrons',
    f'fj_{x}_sj1_nchadrons',
    f'fj_{x}_sj2_nchadrons',
    f'fj_{x}_part_sumpt',
    f'fj_{x}_cpart_sumpt',
    f'fj_{x}_gpart_sumpt',
    f'fj_{x}_cpart_ptratio',
    f'fj_{x}_gpart_ptratio',
]
all_vars = set(train_vars + obs_vars) - set([f'fj_{x}_cpart_ptratio', f'fj_{x}_gpart_ptratio'])

# ---------------------------------------------
def add_vars(df):
    df[f'fj_{x}_gpart_ptratio'] = df.eval(f'fj_{x}_gpart_sumpt/fj_{x}_part_sumpt')
    df[f'fj_{x}_cpart_ptratio'] = df.eval(f'fj_{x}_cpart_sumpt/fj_{x}_part_sumpt')
    return df

def fix_uint64_branches(df):
    for k in df.keys():
        if df[k].dtype in (np.uint32, np.uint64) :
            df[k] = df[k].astype(np.int64)
    return df

def make_dmatrix(filepath='', predict=False, k_folds=5, random_state=None):
    if predict:
        print('...loading', filepath)
        df = fix_uint64_branches(uproot.open(filepath)['Events'].pandas.df(train_vars+['event']))
        print('...done')
        X = df[train_vars]
        outputs = []
        for idx in range(k_folds):
            pos = (df['event'] % k_folds == idx)
            dmat = xgb.DMatrix(X[pos], feature_names=train_vars)
            outputs.append((pos, dmat))
        return df, outputs
    else:
        print('basesel:', basesel)
        tmp_dfs = []
        for basedir in basedirs:
            for proc, f in flist_train:
                print('...loading proc %s file %s' % (proc, os.path.join(basedir, f)))
                df = fix_uint64_branches(uproot.open(os.path.join(basedir, f))['Events'].pandas.df(all_vars))
                df.query(basesel, inplace=True)
                add_vars(df)
                df.query(f'({sigexpr}) | ({bkgexpr})', inplace=True)
                df[label_var] = df.eval(sigexpr)
                df['train_process'] = proc
                df[wgtvar] = df.eval(wgtexpr)
                tmp_dfs.append(df)
        a = pd.concat(tmp_dfs, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

        sigpos = a[label_var] == 1
        bkgpos = a[label_var] == 0
        nsig = sigpos.sum()
        nbkg = bkgpos.sum()
        sigwgtsum = a.loc[sigpos, wgtvar].sum()
        bkgwgtsum = a.loc[bkgpos, wgtvar].sum()
        print('before reweight: sigwgt-sum=%f sig-nevts=%d      bkgwgt-sum=%f bkg-nevts=%d' % (sigwgtsum, nsig, bkgwgtsum, nbkg))

        # reweight signal such that the avg wgt is 1
        a.loc[sigpos, wgtvar] = a.loc[sigpos, wgtvar] * (nsig / sigwgtsum)
        a.loc[bkgpos, wgtvar] = a.loc[bkgpos, wgtvar] * (nsig / bkgwgtsum)

        sigwgtsum = a.loc[sigpos, wgtvar].sum()
        bkgwgtsum = a.loc[bkgpos, wgtvar].sum()
        print('after reweight: sigwgt-sum=%f sig-nevts=%d      bkgwgt-sum=%f bkg-nevts=%d' % (sigwgtsum, nsig, bkgwgtsum, nbkg))
        print(a[:10])

        X = a[train_vars]
        y = a[label_var]
        W = a[wgtvar]

        def out_fn(idx):
            val_pos = (a['event'] % k_folds == idx)
            train_pos = np.logical_not(val_pos)
            d_train = xgb.DMatrix(X[train_pos], y[train_pos], weight=W[train_pos], feature_names=train_vars)
            d_val = xgb.DMatrix(X[val_pos], y[val_pos], weight=W[val_pos], feature_names=train_vars)
            print(X[train_pos][:10], y[train_pos][:10], W[train_pos][:10])
            return d_train, d_val

        return out_fn


def plotROC(y_score, y_true, sample_weight=None):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)

    def remove_unordered(fpr, tpr):
        mask = np.array([True] * len(fpr))
        found = False
        for i in range(1, len(fpr) - 1):
            if fpr[i - 1] > fpr[i] or fpr[i] > fpr[i + 1] or tpr[i - 1] > tpr[i] or tpr[i] > tpr[i + 1]:
                mask[i] = False
                found = True
        return found, fpr[mask], tpr[mask]

    while True:
        found, fpr, tpr = remove_unordered(fpr, tpr)
        if not found or len(fpr) == 0:
            break

    roc_auc = auc(fpr, tpr)

    plt.figure()
    legend = '%s (area = %0.4f)' % ('BDT', roc_auc)
    print(legend)
    plt.plot(tpr, 1 - fpr, label=legend)
    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0, 1])
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background rejection')
    plt.legend(loc='best')
#     plt.yscale('log')
    plt.grid()

    return roc_auc


def train(hyper_params=None):
    model_dir = args.model_dir if hyper_params is None else os.path.join(args.model_dir, 'eta_%.6f_depth_%d' % (hyper_params['eta'], hyper_params['max_depth']))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    shutil.copy(__file__, model_dir)
    model_file = os.path.join(model_dir, model_name)

    # # setup parameters for xgboost
    param = {}
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = ['error', 'auc', 'logloss']
    # param['min_child_weight'] = 1
    # param['gamma']=0.01
    param['eta'] = 0.01
    param['max_depth'] = 4
    # param['colsample_bytree'] = 0.8
    param['subsample'] = 0.8

    # gpu
    if args.gpu:
        param['gpu_id'] = int(args.gpu)
        param['tree_method'] = 'gpu_hist'
        param['max_bin'] = 2048
#         param['predictor'] = 'cpu_predictor'

    if hyper_params is not None:
        param.update(hyper_params)
    print('Starting training...')
    print('xgboost params:')
    print(param)

    num_round = 2000

    y_labels, y_preds, wgts = [], [], []
    get_dmat = make_dmatrix(k_folds=k_folds, random_state=None)

    for idx in range(k_folds):
        print('Training round %d' % idx)

        print('Start loading files...')
        d_train, d_test = get_dmat(idx)

        print('Using training files with %d events...' % (d_train.num_row()))
        print('Using validation files with %d events...' % (d_test.num_row()))

        print('Training vars:\n' + '\n'.join(train_vars))
        print('%d vars in total' % len(train_vars))

        watchlist = [ (d_train, 'train'), (d_test, 'eval') ]

        bst = xgb.train(param, d_train, num_round, watchlist, early_stopping_rounds=20)

        bst.save_model('%s.%d' % (model_file, idx))

        importance_type = 'weight'
        print('Using importance_type=%s' % importance_type)
        scores = bst.get_score(importance_type=importance_type)
        ivar = 1
        for k in sorted(scores, key=scores.get, reverse=True):
            print("%2d. %24s: %s" % (ivar, k, str(scores[k])))
            ivar = ivar + 1

        importance_type = 'gain'
        print('Using importance_type=%s' % importance_type)
        scores = bst.get_score(importance_type=importance_type)
        ivar = 1
        for k in sorted(scores, key=scores.get, reverse=True):
            print("%2d. %24s: %s" % (ivar, k, str(scores[k])))
            ivar = ivar + 1

        importance_type = 'cover'
        print('Using importance_type=%s' % importance_type)
        scores = bst.get_score(importance_type=importance_type)
        ivar = 1
        for k in sorted(scores, key=scores.get, reverse=True):
            print("%2d. %24s: %s" % (ivar, k, str(scores[k])))
            ivar = ivar + 1

        # testing
        del bst
        bst = xgb.Booster({'predictor':'cpu_predictor'})
        bst.load_model('%s.%d' % (model_file, idx))
        y_preds.append(bst.predict(d_test))
        y_labels.append(d_test.get_label())
        wgts.append(d_test.get_weight())

    y_labels = np.concatenate(y_labels)
    y_preds = np.concatenate(y_preds)
    wgts = np.concatenate(wgts)
    auc = plotROC(y_preds, y_labels, sample_weight=wgts)
#    plt.savefig(os.path.join(model_dir, 'roc.pdf'))
    plt.ylim(0.8, 1)
    plt.savefig(os.path.join(model_dir, 'roc_zoom.pdf'))

    return auc


def predict():
    output_dir = args.outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in flist_pred:
        fpath = os.path.join(args.inputdir, f)
        if not os.path.exists(fpath):
            print('Ignore non-existing file: %s' % fpath)
        df, dmats = make_dmatrix(fpath, predict=True, k_folds=k_folds)
        df[args.bdt_varname] = -99 * np.ones(df.shape[0])
        for idx, (pos, dmat) in enumerate(dmats):
            bst = xgb.Booster({'predictor':'cpu_predictor'})
            bst.load_model('%s.%d' % (os.path.join(args.model_dir, model_name), idx))
            df.loc[pos, args.bdt_varname] = bst.predict(dmat)
        assert not np.any(df[args.bdt_varname] == -99)

        outputpath = os.path.join(output_dir, f)
        if not os.path.exists(os.path.dirname(outputpath)):
            os.makedirs(os.path.dirname(outputpath))
        print('Write prediction file to %s' % outputpath)

#         from root_numpy import array2root
#         array2root(df.to_records(index=False), filename=outputpath, treename='Events', mode='RECREATE')

        with uproot.recreate(outputpath, compression=uproot.write.compress.LZ4(4)) as fout:
            fout['Events'] = uproot.newtree({k:df[k].dtype for k in df.keys()})
            step = 2 ** 20
            start = 0
            while start < len(df) - 1:
                fout['Events'].extend({k:df[k][start:start + step].values for k in df.keys()})
                start += step

if __name__ == '__main__':
    if args.train:
        train()
    elif args.predict:
        predict()