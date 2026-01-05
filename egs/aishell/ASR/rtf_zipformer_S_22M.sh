export CUDA_VISIBLE_DEVICES="6"

./zipformer/benchmark_rtf.py \
    --benchmark-mode single_cpu \
    --exp-dir ./zipformer/exp-S-22M \
    --num-batches 200 \
    --epoch 12 \
    --avg 3 \
    --num-encoder-layers 1,1,1,1  \
    --feedforward-dim 256,256,256,256  \
    --encoder-dim 256,512,768,256 \
    --encoder-unmasked-dim 192,256,320,192 \
    --downsampling-factor "1,4,8,2"  \
    --num-heads "4,4,4,4"  \
    --cnn-module-kernel "31,15,15,31"  \
    --decoding-method modified_beam_search \
    --beam-size 4