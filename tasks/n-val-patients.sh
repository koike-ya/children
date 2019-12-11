for nval in 1 2 3 4 5
do
    exptid="nval-${nval}"
    exptids="${exptids},${exptid}"
#    /home/cs11/.conda/envs/brain/bin/python experiment.py --manifest-path /media/cs11/C420894C20894700/children/WJ01003H_manifest.csv --loss-weight 1.0,1.0 --epochs 15 --epoch-rate 0.6 --batch-size 64 --window-size 2 --window-stride 1 --model-type cnn_rnn --lr 0.0001 --no-inference-softmax --cv-type patient --spect --data-type children --high-cutoff 120 --k-fold 9 --model-path output/n-val.pth --cuda --silent --expt-id ${exptid} --n-val-patients ${nval}
#        cp /media/cs11/Storage/koike/children/output/metrics/muscle-noise-0.0_val.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_val.csv
#        cp /media/cs11/Storage/koike/children/output/metrics/muscle-noise-0.0_test.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_test.csv
done
/home/cs11/.conda/envs/brain/bin/python visualize.py --expt-ids ${exptids:1} --expt-name n-val-patients

