export CUDA_VISIBLE_DEVICES="6,7"

./pruned_transducer_stateless3/train.py \
    --world-size 2 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir pruned_transducer_stateless3/exp-M-40M \
    --use-fp16 1 \
    --num-encoder-layers 14 \
    --dim-feedforward 1280 \
    --nhead 4 \
    --encoder-dim 320 \
    --max-duration 300