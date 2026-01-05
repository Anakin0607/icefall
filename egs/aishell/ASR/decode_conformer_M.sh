export CUDA_VISIBLE_DEVICES="6" # use gpu in this list, in torch.device(0) is the first device in this list

./pruned_transducer_stateless3/decode.py \
    --epoch 16 \
    --avg 3 \
    --exp-dir ./pruned_transducer_stateless3/exp-M-40M \
    --lang-dir data/lang_char \
    --max-duration 1200 \
    --num-encoder-layers 14 \
    --dim-feedforward 1280 \
    --nhead 4 \
    --encoder-dim 320 \
    --decoding-method modified_beam_search \
    --beam-size 4