export WORKDIR=`pwd`
cd $1
python ${WORKDIR}/write_cards_sf.py $1

echo "+++ Converting datacard to workspace +++"
text2workspace.py -m 125 -P HiggsAnalysis.CombinedLimit.TagAndProbeExtendedV2:tagAndProbe SF.txt --PO categories=flvC,flvB,flvL
echo "+++ Fitting... +++"
combine -M MultiDimFit -m 125 SF.root --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5. | tee fit.log
combine -M FitDiagnostics -m 125 SF.root --saveShapes --saveWithUncertainties --robustFit=1 --cminDefaultMinimizerTolerance 5. > /dev/null 2>&1
if [ -n "$2" -a "$2" == 'full' ]; then # run impact
    combineTool.py -M Impacts -d SF.root -m 125 --doInitialFit --robustFit 1 >> pdf.log 2>&1
    combineTool.py -M Impacts -d SF.root -m 125 --robustFit 1 --doFits >> pdf.log 2>&1
    combineTool.py -M Impacts -d SF.root -m 125 -o impacts.json >> pdf.log 2>&1
    plotImpacts.py -i impacts.json -o impacts >> pdf.log 2>&1
fi
cd -