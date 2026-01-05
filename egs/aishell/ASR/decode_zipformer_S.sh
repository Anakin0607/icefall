export CUDA_VISIBLE_DEVICES="6" # use gpu in this list, in torch.device(0) is the first device in this list

./zipformer/decode.py \
    --epoch 12 \
    --avg 3 \
    --exp-dir ./zipformer/exp-S-22M-1 \
    --lang-dir data/lang_char \
    --max-duration 600 \
    --num-encoder-layers 1,1,1,1  \
    --feedforward-dim 256,256,256,256  \
    --encoder-dim 256,512,768,256 \
    --encoder-unmasked-dim 192,256,320,192 \
    --downsampling-factor "1,2,4,2"  \
    --num-heads "4,4,4,4"  \
    --cnn-module-kernel "31,15,15,31"  \
    --decoding-method modified_beam_search \
    --beam-size 4