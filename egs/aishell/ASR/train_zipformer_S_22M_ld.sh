export CUDA_VISIBLE_DEVICES="4,6"

./zipformer/train.py \
  --world-size 1 \
  --num-epochs 12 \
  --start-epoch 1 \
  --exp-dir zipformer/exp-S-22M-1 \
  --use-fp16 1 \
  --num-encoder-layers 1,1,1,1  \
  --feedforward-dim 256,256,256,256  \
  --encoder-dim 256,512,768,256 \
  --encoder-unmasked-dim 192,256,320,192 \
  --downsampling-factor "1,2,4,2"  \
  --num-heads "4,4,4,4"  \
  --cnn-module-kernel "31,15,15,31"  \
  --lr-epochs 1.5 \
  --max-duration 500