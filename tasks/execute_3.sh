for aug in muscle eye white gain none
do
    for pat in chb01 chb02 chb03
    do
        exptid="${pat}-${aug}"
        /home/cs11/.conda/envs/brain/bin/python experiment.py --manifest-path /media/cs11/Storage/koike/chb-mit/${pat}/interictal_preictal/manifest.csv --loss-weight 1.0,1.0 --epochs 20 --epoch-rate 1.0 --batch-size 32 --window-size 2 --window-stride 1 --model-type cnn_rnn --lr 0.00005 --no-inference-softmax --cv-type ictal --spect --data-type chbmit --only-one-patient --high-cutoff 120 --model-path output/${pat}.pth --expt-id ${exptid} --augment ${aug} --cuda
    done
done
