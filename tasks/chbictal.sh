#for aug in muscle-noise eye-noise white-noise shift-gain spec-augment
for aug in muscle-noise
do
    exptids=""
    for value in 0.0
    do
        patids=""
        for pat in chb01 chb02 chb03 chb10 chb14 chb20 chb21 chb22 chb23
        do
            exptid="${pat}-${aug}-${value}"
            patids="${patids},${exptid}"
#        /home/cs11/.conda/envs/brain/bin/python experiment.py --manifest-path /media/cs11/Storage/koike/chb-mit/${pat}/interictal_preictal/manifest.csv --loss-weight 1.0,1.0 --epochs 10 --epoch-rate 1.0 --batch-size 32 --window-size 2 --window-stride 1 --model-type cnn_rnn --lr 0.0001 --no-inference-softmax --cv-type ictal --spect --data-type chbmit --only-one-patient --high-cutoff 120 --model-path output/${pat}.pth --expt-id ${exptid} --${aug} ${value} --cuda
            cp /home/tomoya/workspace/research/brain/children/output/metrics/${pat}-crnn-ictal_val.csv /home/tomoya/workspace/research/brain/children/output/metrics/${exptid}_val.csv
#            cp /home/tomoya/workspace/research/brain/children/output/metrics/${pat}-crnn-ictal_test.csv /home/tomoya/workspace/research/brain/children/output/metrics/${exptid}_test.csv
#        /home/cs11/.conda/envs/brain/bin/python aggregate.py --expt-ids ${patids:1} --expt-name ${aug}-${value}-ictal
        exptids="${exptids},${aug}-${value}-chbictal"
        done
    done
#    /home/cs11/.conda/envs/brain/bin/python visualize.py --expt-ids ${exptids:1} --expt-name ${aug}-cvictal
done
