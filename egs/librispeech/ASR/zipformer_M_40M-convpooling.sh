export CUDA_VISIBLE_DEVICES="6,7"

# ./zipformer/train_kvpooling.py \
#   --world-size 2 \
#   --num-epochs 20 \
#   --start-epoch 1 \
#   --bpe-model data/lang_bpe_5000/bpe.model \
#   --exp-dir zipformer/exp-M-40M-convpooling-42111 \
#   --use-fp16 1 \
#   --num-encoder-layers  2,2,3,3,2 \
#   --feedforward-dim 512,768,1024,1024,768  \
#   --encoder-dim 192,256,384,384,256 \
#   --encoder-unmasked-dim 192,192,256,256,192 \
#   --downsampling-factor "1,2,4,4,2"  \
#   --num-heads "4,4,4,4,4"  \
#   --cnn-module-kernel "31,31,15,15,31"  \
#   --pooling-mode "conv" \
#   --pooling-stride "4,2,1,1,1" \
#   --lr-epochs 1.5 \
#   --master-port 12357 \
#   --max-duration 600 # total audio length in one batch

./zipformer/decode_kvpooling.py \
    --epoch 20 \
    --avg 3 \
    --exp-dir ./zipformer/exp-M-40M-convpooling-42111 \
    --lang-dir data/lang_bpe_5000 \
    --bpe-model data/lang_bpe_5000/bpe.model \
    --max-duration 1200 \
    --num-encoder-layers  2,2,3,3,2 \
    --feedforward-dim 512,768,1024,1024,768  \
    --encoder-dim 192,256,384,384,256 \
    --encoder-unmasked-dim 192,192,256,256,192 \
    --downsampling-factor "1,2,4,4,2"  \
    --num-heads "4,4,4,4,4"  \
    --cnn-module-kernel "31,31,15,15,31"  \
    --pooling-mode "conv" \
    --pooling-stride "4,2,1,1,1" \
    --decoding-method modified_beam_search \
    --beam-size 4
