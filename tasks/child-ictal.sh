#for aug in muscle-noise eye-noise white-noise shift-gain
for aug in spec-augment
do
    exptids=""
    for value in 0.0 0.2 0.4
    do
        patids=""
        for pat in YJ0100DP YJ0112PQ MJ00803P WJ010024 YJ0100E9
#        for pat in MJ00803P
        do
            exptid="${pat}-${aug}-${value}"
            patids="${patids},${exptid}"
#            /home/cs11/.conda/envs/brain/bin/python experiment.py --manifest-path /media/cs11/C420894C20894700/children/${pat}_manifest.csv --loss-weight 1.0,1.0 --epochs 1 --epoch-rate 1.0 --batch-size 32 --n-use-eeg 1 --sample-rate 500 --window-size 2 --window-stride 1 --model-type cnn_rnn --lr 0.0001 --k-fold 9 --no-inference-softmax --sample-balance 1.0,1.0 --cv-type ictal --spect --only-one-patient --model-path output/${pat}-cvictal.pth --expt-id ${exptid} --${aug} ${value} --cuda
#            cp /media/cs11/Storage/koike/children/output/metrics/${pat}-muscle-noise-0.0_val.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_val.csv
#            cp /media/cs11/Storage/koike/children/output/metrics/${pat}-muscle-noise-0.0_test.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_test.csv
        done
        /home/cs11/.conda/envs/brain/bin/python aggregate.py --expt-ids ${patids:1} --expt-name ${aug}-${value}-ictal
        exptids="${exptids},${aug}-${value}-ictal"
    done
    /home/cs11/.conda/envs/brain/bin/python visualize.py --expt-ids ${exptids:1} --expt-name ${aug}-cvictal
done
