export CUDA_VISIBLE_DEVICES="6"

./pruned_transducer_stateless3/benchmark_rtf.py \
    --benchmark-mode single_cpu \
    --exp-dir ./pruned_transducer_stateless3/exp-S-20M \
    --num-batches 20 \
    --epoch 16 \
    --avg 3 \
    --num-encoder-layers 12 \
    --dim-feedforward 800 \
    --nhead 4 \
    --encoder-dim 200 \
    --decoding-method modified_beam_search \
    --beam-size 4