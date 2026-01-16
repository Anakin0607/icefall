export CUDA_VISIBLE_DEVICES="6,7"

# ./zipformer/train_kvpooling.py \
#   --world-size 2 \
#   --num-epochs 20 \
#   --start-epoch 12 \
#   --exp-dir zipformer/exp-S-20M-convpooling \
#   --use-fp16 1 \
#   --num-encoder-layers "2,2,3,2,2" \
#   --downsampling-factor "1,2,4,2,1" \
#   --encoder-dim "128,192,256,192,128" \
#   --encoder-unmasked-dim "96,128,192,128,96" \
#   --feedforward-dim "384,576,768,576,384" \
#   --num-heads "4,4,4,4,4" \
#   --cnn-module-kernel "31,31,15,31,31" \
#   --pooling-mode "conv" \
#   --pooling-stride "4,2,1,1,1" \
#   --lr-epochs 1.5 \
#   --max-duration 500

# decode
./zipformer/decode_pooling.py \
  --epoch 10 \
  --avg 2 \
  --exp-dir zipformer/exp-S-20M-convpooling \
  --lang-dir data/lang_char \
  --num-encoder-layers "2,2,3,2,2" \
  --downsampling-factor "1,2,4,2,1" \
  --encoder-dim "128,192,256,192,128" \
  --encoder-unmasked-dim "96,128,192,128,96" \
  --feedforward-dim "384,576,768,576,384" \
  --num-heads "4,4,4,4,4" \
  --cnn-module-kernel "31,31,15,31,31" \
  --pooling-mode "conv" \
  --pooling-stride "4,2,1,1,1" \
  --decoding-method modified_beam_search \
  --max-duration 1200 # total audio length in one batch