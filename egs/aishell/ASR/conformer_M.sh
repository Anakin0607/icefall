export CUDA_VISIBLE_DEVICES="6,7"

# train
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

# decode
# ./pruned_transducer_stateless3/decode.py \
#     --epoch 16 \
#     --avg 3 \
#     --exp-dir ./pruned_transducer_stateless3/exp-M-40M \
#     --lang-dir data/lang_char \
#     --max-duration 1200 \
#     --num-encoder-layers 14 \
#     --dim-feedforward 1280 \
#     --nhead 4 \
#     --encoder-dim 320 \
#     --decoding-method modified_beam_search \
#     --beam-size 4

# test_rtf
# ./pruned_transducer_stateless3/benchmark_rtf.py \
#     --benchmark-mode single_cpu \
#     --exp-dir ./pruned_transducer_stateless3/exp-M-40M \
#     --num-batches 20 \
#     --epoch 16 \
#     --avg 3 \
#     --num-encoder-layers 14 \
#     --dim-feedforward 1280 \
#     --nhead 4 \
#     --encoder-dim 320 \
#     --decoding-method modified_beam_search \
#     --beam-size 4