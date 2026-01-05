export CUDA_VISIBLE_DEVICES="6" # use gpu in this list, in torch.device(0) is the first device in this list

./pruned_transducer_stateless3/decode.py \
    --epoch 17 \
    --avg 4 \
    --exp-dir ./pruned_transducer_stateless3/exp-S-27M \
    --lang-dir data/lang_char \
    --max-duration 1200 \
    --num-encoder-layers 12 \
    --dim-feedforward 1024 \
    --nhead 4 \
    --encoder-dim 256 \
    --decoding-method modified_beam_search \
    --beam-size 4