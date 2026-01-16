# KV-pooling zipformer M scale, is S scale to original zipformer papaer

export CUDA_VISIBLE_DEVICES="0,1"

# ./zipformer/train.py \
#   --world-size 2 \
#   --num-epochs 12 \
#   --start-epoch 1 \
#   --exp-dir zipformer/exp-M-40M \
#   --use-fp16 1 \
#   --num-encoder-layers  2,2,3,3,2 \
#   --feedforward-dim 512,768,1024,1024,768  \
#   --encoder-dim 192,256,384,384,256 \
#   --encoder-unmasked-dim 192,192,256,256,192 \
#   --downsampling-factor "1,2,4,4,2"  \
#   --num-heads "4,4,4,4,4"  \
#   --cnn-module-kernel "31,31,15,15,31"  \
#   --lr-epochs 1.5 \
#   --max-duration 300 # total audio length in one batch

# decode
# ./zipformer/decode.py \
#     --epoch 12 \
#     --avg 3 \
#     --exp-dir ./zipformer/exp-M-40M \
#     --lang-dir data/lang_char \
#     --max-duration 600 \
#     --num-encoder-layers  2,2,3,3,2 \
#     --feedforward-dim 512,768,1024,1024,768  \
#     --encoder-dim 192,256,384,384,256 \
#     --encoder-unmasked-dim 192,192,256,256,192 \
#     --downsampling-factor "1,2,4,4,2"  \
#     --num-heads "4,4,4,4,4"  \
#     --cnn-module-kernel "31,31,15,15,31"  \
#     --decoding-method modified_beam_search \
#     --beam-size 4

# test_rtf
# ./zipformer/benchmark_rtf.py \
#     --benchmark-mode single_cpu \
#     --exp-dir ./zipformer/exp-M-40M \
#     --num-batches 20 \
#     --epoch 12 \
#     --avg 3 \
#     --num-encoder-layers  2,2,3,3,2 \
#     --feedforward-dim 512,768,1024,1024,768  \
#     --encoder-dim 192,256,384,384,256 \
#     --encoder-unmasked-dim 192,192,256,256,192 \
#     --downsampling-factor "1,2,4,4,2"  \
#     --num-heads "4,4,4,4,4"  \
#     --cnn-module-kernel "31,31,15,15,31"  \
#     --decoding-method modified_beam_search \
#     --beam-size 4

python ./zipformer/visualize_attn.py \
    --epoch 12 \
    --avg 3 \
    --exp-dir ./zipformer/exp-M-40M \
    --wav-path data/DEV_T0000000000.wav \
    --save-dir ./zipformer/attention_analysis \
    --num-encoder-layers  2,2,3,3,2 \
    --feedforward-dim 512,768,1024,1024,768  \
    --encoder-dim 192,256,384,384,256 \
    --encoder-unmasked-dim 192,192,256,256,192 \
    --downsampling-factor "1,2,4,4,2"  \
    --num-heads "4,4,4,4,4"  \
    --cnn-module-kernel "31,31,15,15,31"  \
    --max-duration 600 \
    --num-batches 1
