# KV-pooling zipformer M scale, is S scale to original zipformer papaer

export CUDA_VISIBLE_DEVICES="0,1"

./zipformer/train_kvpooling.py \
  --world-size 2 \
  --num-epochs 20 \
  --start-epoch 1 \
  --exp-dir zipformer/exp-M-40M-convpooling-42111-lowlr \
  --use-fp16 1 \
  --num-encoder-layers  2,2,3,3,2 \
  --feedforward-dim 512,768,1024,1024,768  \
  --encoder-dim 192,256,384,384,256 \
  --encoder-unmasked-dim 192,192,256,256,192 \
  --downsampling-factor "1,2,4,4,2"  \
  --num-heads "4,4,4,4,4"  \
  --cnn-module-kernel "31,31,15,15,31"  \
  --pooling-mode "conv" \
  --pooling-stride "4,2,1,1,1" \
  --lr-epochs 1.5 \
  --master-port 12356 \
  --max-duration 600 # total audio length in one batch

#decode
# ./zipformer/decode_pooling.py \
#   --epoch 14 \
#   --avg 3 \
#   --exp-dir zipformer/exp-M-40M-convpooling-42111-lowlr \
#   --lang-dir data/lang_char \
#   --num-encoder-layers  2,2,3,3,2 \
#   --feedforward-dim 512,768,1024,1024,768  \
#   --encoder-dim 192,256,384,384,256 \
#   --encoder-unmasked-dim 192,192,256,256,192 \
#   --downsampling-factor "1,2,4,4,2"  \
#   --num-heads "4,4,4,4,4"  \
#   --cnn-module-kernel "31,31,15,15,31"  \
#   --pooling-mode "conv" \
#   --pooling-stride "4,2,1,1,1" \
#   --decoding-method modified_beam_search \
#   --max-duration 1200 # total audio length in one batch

# rtf test
# ./zipformer/benchmark_rtf_pooling.py \
#    --benchmark-mode single_cpu \
#    --exp-dir ./zipformer/exp-M-40M-convpooling-42111 \
#    --num-batches 200 \
#    --epoch 11 \
#    --avg 3 \
#    --num-encoder-layers  2,2,3,3,2 \
#    --feedforward-dim 512,768,1024,1024,768  \
#    --encoder-dim 192,256,384,384,256 \
#    --encoder-unmasked-dim 192,192,256,256,192 \
#    --downsampling-factor "1,2,4,4,2"  \
#    --num-heads "4,4,4,4,4"  \
#    --cnn-module-kernel "31,31,15,15,31"  \
#    --pooling-mode "conv" \
#    --pooling-stride "4,2,1,1,1" \
#    --decoding-method modified_beam_search \
#    --beam-size 4

./zipformer/visualize_attn.py \
    --epoch 11 \
    --avg 3 \
    --exp-dir ./zipformer/exp-M-40M-convpooling-42111 \
    --chunk-size 64 \
    --wav-path data/DEV_T0000000000.wav \
    --save-dir ./zipformer/attention_analysis \
    --num-encoder-layers  2,2,3,3,2 \
    --feedforward-dim 512,768,1024,1024,768  \
    --encoder-dim 192,256,384,384,256 \
    --encoder-unmasked-dim 192,192,256,256,192 \
    --downsampling-factor "1,2,4,4,2"  \
    --num-heads "4,4,4,4,4"  \
    --cnn-module-kernel "31,31,15,15,31"  \
    --pooling-mode "conv" \
    --pooling-stride "4,2,1,1,1" \
    --max-duration 300 \
    --num-batches 1