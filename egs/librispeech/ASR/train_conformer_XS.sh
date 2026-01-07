export CUDA_VISIBLE_DEVICES="4"

./pruned_transducer_stateless5/train.py \
    --world-size 1 \
    --num-epochs 16 \
    --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-S-20M \
    --use-fp16 1 \
    --num-encoder-layers 12 \
    --dim-feedforward 1024 \
    --nhead 4 \
    --encoder-dim 256 \
    --max-duration 900