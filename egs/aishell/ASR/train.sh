python ./zipformer_adapter_gate/train.py  \
  --world-size 1 \
  --num-epochs 10  \
  --start-epoch 1  \
  --exp-dir zipformer_adapter_gate/exp_dyn \
  --use-adapters True \
  --adapter-dim 16  \
  --do-finetune 1  \
  --use-fp16 1 \
  --use-mux 0 \
  --finetune-ckpt zipformer_adapter_gate/exp_dyn/epoch-40.pt \
  --save-every-n 10000000 \
  --max-duration 300 \
  --num-encoder-layers 1,1,1,1  \
  --feedforward-dim 256,256,256,256  \
  --encoder-dim 256,512,768,256 \
  --encoder-unmasked-dim 192,256,320,192 \
  --downsampling-factor "1,4,8,2"  \
  --num-heads "4,4,4,4"  \
  --cnn-module-kernel "31,15,15,31"  \
  --base-lr 0.0045 \
  --master-port 22558

python ./zipformer_adapter_gate/decode.py  \
  --epoch 10 \
  --avg 1 \
  --exp-dir zipformer_adapter_gate/exp_dyn \
  --use-adapters True \
  --adapter-dim 16  \
  --max-duration 600 \
  --num-encoder-layers 1,1,1,1  \
  --feedforward-dim 256,256,256,256  \
  --encoder-dim 256,512,768,256 \
  --encoder-unmasked-dim 192,256,320,192 \
  --downsampling-factor "1,4,8,2"  \
  --num-heads "4,4,4,4"  \
  --cnn-module-kernel "31,15,15,31"  \
  --use-averaged-model False \
  --decoding-method greedy_search

#40m
  python ./zipformer_adapter/decode.py  \
  --epoch 30 \
  --avg 1 \
  --use-averaged-model False \
  --use-adapters False \
  --exp-dir zipformer_adapter/exp_base_40m \
  --max-duration 600 \
  --num-encoder-layers  2,2,3,3,2 \
  --feedforward-dim 512,768,1024,1024,768  \
  --encoder-dim 192,256,384,384,256 \
  --encoder-unmasked-dim 192,192,256,256,192 \
  --downsampling-factor "1,2,4,4,2"  \
  --num-heads "4,4,4,4,4"  \
  --cnn-module-kernel "31,31,15,15,31"  \
  --decoding-method greedy_search

python ./zipformer_adapter/decode.py  \
  --epoch 40 \
  --avg 1 \
  --use-averaged-model False \
  --exp-dir zipformer_half/exp_base \
  --max-duration 600 \
  --num-encoder-layers 1,1,1,1  \
  --feedforward-dim 256,256,256,256  \
  --encoder-dim 256,512,768,256 \
  --encoder-unmasked-dim 192,256,320,192 \
  --downsampling-factor "1,4,8,2"  \
  --num-heads "4,4,4,4"  \
  --cnn-module-kernel "31,15,15,31"  \
  --decoding-method greedy_search