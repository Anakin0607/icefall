export CUDA_VISIBLE_DEVICES="6"

./zipformer/benchmark_rtf.py \
    --benchmark-mode single_cpu \
    --exp-dir ./zipformer/exp-S-20M \
    --num-batches 200 \
    --epoch 12 \
    --avg 3 \
    --num-encoder-layers "2,2,3,2,2" \
    --downsampling-factor "1,2,4,2,1" \
    --encoder-dim "128,192,256,192,128" \
    --encoder-unmasked-dim "96,128,192,128,96" \
    --feedforward-dim "384,576,768,576,384" \
    --num-heads "4,4,4,4,4" \
    --cnn-module-kernel "31,31,15,31,31" \
    --decoding-method modified_beam_search \
    --beam-size 4