exptids=""
for nlayer in 1 2 3 4 5
do
    exptid="rnn-${nlayer}-layers"
    exptids="${exptids},${exptid}"
    echo $exptids
    #/home/cs11/.conda/envs/brain/bin/python experiment.py --manifest-path /media/cs11/C420894C20894700/children/MJ00802S_manifest.csv --loss-weight 1.0,1.0 --epochs 20 --epoch-rate 1.0 --batch-size 32 --window-size 2 --window-stride 1 --model-type cnn_rnn --lr 0.0001 --no-inference-softmax --cv-type patient --spect --data-type children --high-cutoff 120 --k-fold 9 --model-path output/rnn-layer.pth --cuda --expt-id ${exptid} --rnn-n-layers ${nlayer}
done
/home/cs11/.conda/envs/brain/bin/python visualize.py --expt-ids ${exptids:1} --expt-name rnn-layers
