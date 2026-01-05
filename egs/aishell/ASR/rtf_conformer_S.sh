export CUDA_VISIBLE_DEVICES="6"

./pruned_transducer_stateless3/benchmark_rtf.py \
    --benchmark-mode single_cpu \
    --exp-dir ./pruned_transducer_stateless3/exp-S-27M \
    --num-batches 20 \
    --epoch 17 \
    --avg 3 \
    --num-encoder-layers 12 \
    --dim-feedforward 1024 \
    --nhead 4 \
    --encoder-dim 256 \
    --decoding-method modified_beam_search \
    --beam-size 4