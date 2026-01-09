export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless5/train.py \
    --world-size 4 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-M-40M \
    --use-fp16 1 \
    --num-encoder-layers 14 \
    --dim-feedforward 1280 \
    --nhead 4 \
    --encoder-dim 320 \
    --max-duration 300 \
    --master-port 12300 \
    --start-batch 432000