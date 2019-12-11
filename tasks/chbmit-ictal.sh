#for aug in muscle-noise eye-noise white-noise shift-gain spec-augment
for aug in white-noise
do
    exptids=""
    for value in 0.0
    do
        patids=""
        for pat in chb21 chb23
        do
            exptid="${pat}-${aug}-${value}"
            patids="${patids},${exptid}"
            python experiment.py --manifest-path /media/cs11/Storage/koike/chb-mit/${pat}/interictal_preictal/manifest.csv --loss-weight 1.0,1.0 --epochs 10 --epoch-rate 0.8 --batch-size 32 --window-size 2 --window-stride 1 --model-type cnn_rnn --lr 0.0001 --no-inference-softmax --expt-id crnn-chbmit --cv-type ictal --spect --data-type chbmit --only-one-patient --high-cutoff 120 --cuda --model-path ../output/pat-cnnrnn.pth --expt-id ${exptid} --${aug} ${value}
#            cp /media/cs11/Storage/koike/children/output/metrics/${pat}-muscle-noise-0.0_val.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_val.csv
#            cp /media/cs11/Storage/koike/children/output/metrics/${pat}-muscle-noise-0.0_test.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_test.csv
        done
#        /home/cs11/.conda/envs/brain/bin/python aggregate.py --expt-ids ${patids:1} --expt-name ${aug}-${value}-ictal
        exptids="${exptids},${aug}-${value}-ictal"
    done
#    /home/cs11/.conda/envs/brain/bin/python visualize.py --expt-ids ${exptids:1} --expt-name ${aug}-cvictal
done
