# KV-pooling zipformer M scale, is S scale to original zipformer papaer

export CUDA_VISIBLE_DEVICES="6"

./zipformer/decode_pooling.py \
  --epoch 17 \
  --avg 3 \
  --exp-dir zipformer/exp-M-40M-convpooling \
  --lang-dir data/lang_char \
  --num-encoder-layers  2,2,3,3,2 \
  --feedforward-dim 512,768,1024,1024,768  \
  --encoder-dim 192,256,384,384,256 \
  --encoder-unmasked-dim 192,192,256,256,192 \
  --downsampling-factor "1,2,4,4,2"  \
  --num-heads "4,4,4,4,4"  \
  --cnn-module-kernel "31,31,15,15,31"  \
  --pooling-mode "conv" \
  --pooling-stride "4,2,1,1,2" \
  --decoding-method modified_beam_search \
  --max-duration 600 # total audio length in one batch