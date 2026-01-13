export CUDA_VISIBLE_DEVICES="4"

./zipformer/train.py \
  --world-size 1 \
  --num-epochs 20 \
  --start-epoch 1 \
  --exp-dir zipformer/exp-S-20M-bpe5000 \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --use-fp16 1 \
  --num-encoder-layers "2,2,3,2,2" \
  --downsampling-factor "1,2,4,2,1" \
  --encoder-dim "128,192,256,192,128" \
  --encoder-unmasked-dim "96,128,192,128,96" \
  --feedforward-dim "384,576,768,576,384" \
  --num-heads "4,4,4,4,4" \
  --cnn-module-kernel "31,31,15,31,31" \
  --lr-epochs 1.5 \
  --max-duration 900