export CUDA_VISIBLE_DEVICES="6"

./zipformer/benchmark_rtf.py \
    --benchmark-mode single_cpu \
    --exp-dir ./zipformer/exp-M-40M \
    --num-batches 20 \
    --epoch 12 \
    --avg 3 \
    --num-encoder-layers  2,2,3,3,2 \
    --feedforward-dim 512,768,1024,1024,768  \
    --encoder-dim 192,256,384,384,256 \
    --encoder-unmasked-dim 192,192,256,256,192 \
    --downsampling-factor "1,2,4,4,2"  \
    --num-heads "4,4,4,4,4"  \
    --cnn-module-kernel "31,31,15,15,31"  \
    --decoding-method modified_beam_search \
    --beam-size 4