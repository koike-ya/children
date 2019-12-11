for w-size in 5 10 15
#for aug in white-noise
do
    exptids=""
    for w-stride in 2 5 10
    do
        exptid="w-size-${w-size}-w-stride-${w-stride}"
        exptids="${exptids},${exptid}"
        /home/cs11/.conda/envs/brain/bin/python experiment.py --manifest-path /media/cs11/C420894C20894700/children/WJ01003H_manifest.csv --loss-weight 1.0,1.0 --epochs 15 --epoch-rate 0.6 --batch-size 64 --window-size 2 --window-stride 1 --model-type cnn_rnn --lr 0.0001 --no-inference-softmax --cv-type patient --spect --data-type children --high-cutoff 120 --k-fold 9 --model-path output/aug-values.pth --cuda --silent --n-val-patients 4 --expt-id ${exptid} --${aug} ${value}
#        cp /media/cs11/Storage/koike/children/output/metrics/nval-4_val.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_val.csv
#        cp /media/cs11/Storage/koike/children/output/metrics/nval-4_test.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_test.csv
    done
    /home/cs11/.conda/envs/brain/bin/python visualize.py --expt-ids ${exptids:1} --expt-name ${aug}-n-val-4
done
