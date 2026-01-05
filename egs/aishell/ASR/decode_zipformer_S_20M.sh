export CUDA_VISIBLE_DEVICES="6" # use gpu in this list, in torch.device(0) is the first device in this list

./zipformer/decode.py \
    --epoch 12 \
    --avg 3 \
    --exp-dir ./zipformer/exp-S-20M \
    --lang-dir data/lang_char \
    --max-duration 600 \
    --num-encoder-layers "2,2,3,2,2" \
    --downsampling-factor "1,2,4,2,1" \
    --encoder-dim "128,192,256,192,128" \
    --encoder-unmasked-dim "96,128,192,128,96" \
    --feedforward-dim "384,576,768,576,384" \
    --num-heads "4,4,4,4,4" \
    --cnn-module-kernel "31,31,15,31,31" \
    --decoding-method modified_beam_search \
    --beam-size 4