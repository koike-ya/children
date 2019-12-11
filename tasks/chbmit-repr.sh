patids=""
for pat in chb02 chb21 chb23
do
    exptid="${pat}-chbcnn"
    patids="${patids},${exptid}"
    /home/cs11/.conda/envs/brain/bin/python experiment.py --manifest-path /media/cs11/Storage/koike/chb-mit/${pat}/interictal_preictal/manifest.csv --loss-weight 1.0,1.0 --epochs 50 --epoch-rate 1.0 --batch-size 50 --window-size 2 --window-stride 1 --model-type cnn_rnn --lr 0.0001 --no-inference-softmax --cv-type ictal --spect --data-type chbmit --reproduce chbmit-cnn --high-cutoff 120 --k-fold 9 --model-path output/chbcnn.pth --cuda --silent --only-one-patient --model-manager keras --expt-id ${exptid}
#            cp /media/cs11/Storage/koike/children/output/metrics/${pat}-muscle-noise-0.0_val.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_val.csv
#            cp /media/cs11/Storage/koike/children/output/metrics/${pat}-muscle-noise-0.0_test.csv /media/cs11/Storage/koike/children/output/metrics/${exptid}_test.csv
#        /home/cs11/.conda/envs/brain/bin/python aggregate.py --expt-ids ${patids:1} --expt-name ${aug}-${value}-ictal
#    exptids="${exptids},${aug}-${value}-ictal"
#    /home/cs11/.conda/envs/brain/bin/python visualize.py --expt-ids ${exptids:1} --expt-name ${aug}-cvictal
done
