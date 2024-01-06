CWD=/Users/seanryan/Documents/School/Graduate/Thesis/code-base/model/audiogene_ml/audiogene-ml
SCRIPT=$CWD/src/scripts/r-dir-util/plot-audiogram-shape-rule-extra-args.R
DATA=$CWD/data/processed/master_interp_fixed_revise_remove_unknown_loci_name_fix.csv
OUTPUT_DIR=$CWD/output/visualizations/audiogram_shapes/
#for nBins in {1..2..1}
#do
#  for binSize in {20..30..10}
#  do
#    for t1 in {10..35..5}
#    do
#      for t2 in {5..30..5}
#      do
#        if [ $t1 -gt $t2 ]
#        then
#          /Users/seanryan/opt/anaconda3/envs/data_audiogene/bin/R -f $SCRIPT --args 2 $binSize $t1 $t2 $DATA $OUTPUT_DIR "min" "max" $nBins
#        fi
#      done
#    done
#  done
#done
nBins=2
binSize=30
for t1 in {10..35..5}
do
  for t2 in {5..30..5}
  do
    if [ $t1 -gt $t2 ]
    then
      /Users/seanryan/opt/anaconda3/envs/data_audiogene/bin/R -f $SCRIPT --args 2 $binSize $t1 $t2 $DATA $OUTPUT_DIR "min" "max" $nBins
    fi
  done
done
