export CUDA_VISIBLE_DEVICES="2,3"

./pruned_transducer_stateless5/train.py \
    --world-size 2 \
    --num-epochs 30 \
    --start-epoch 1 \
    --bpe-model data/lang_bpe_5000/bpe.model \
    --exp-dir pruned_transducer_stateless5/exp-XS-13M-bpe5000 \
    --use-fp16 1 \
    --num-encoder-layers 16 \
    --dim-feedforward 768 \
    --nhead 4 \
    --encoder-dim 192 \
    --max-duration 900