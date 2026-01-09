export CUDA_VISIBLE_DEVICES="6" # use gpu in this list, in torch.device(0) is the first device in this list

./pruned_transducer_stateless5/decode.py \
    --epoch 30 \
    --avg 4 \
    --exp-dir ./pruned_transducer_stateless5/exp-S-20M \
    --lang-dir data/lang_bpe_500 \
    --max-duration 1200 \
    --num-encoder-layers 12 \
    --dim-feedforward 1024 \
    --nhead 4 \
    --encoder-dim 256 \
    --decoding-method modified_beam_search \
    --beam-size 4