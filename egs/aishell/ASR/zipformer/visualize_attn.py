#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np

from icefall.lexicon import Lexicon
import torchaudio
import torchaudio.compliance.kaldi as kaldi


from icefall.utils import (
    AttributeDict,
    make_pad_mask,
    setup_logger,
    str2bool,
)
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from train_kvpooling import add_model_arguments, get_model, get_params

# 假设你的数据加载模块在这里，如果不是请根据实际情况修改 import
from asr_datamodule import AishellAsrDataModule

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_char",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - modified_beam_search
          - fast_beam_search
          - fast_beam_search_LG
          - fast_beam_search_nbest_oracle
        If you use fast_beam_search_LG, you have to specify
        `--lang-dir`, which should contain `LG.pt`.
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search,
        fast_beam_search, fast_beam_search_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--ilme-scale",
        type=float,
        default=0.2,
        help="""
        Used only when --decoding_method is fast_beam_search_LG.
        It specifies the scale for the internal language model estimation.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search, fast_beam_search_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search, fast_beam_search_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )
    parser.add_argument(
        "--wav-path",
        type=str,
        default="data/DEV_T0000000000.wav",
        help="Path of wav to decode",
    )

    # 绘图输出相关参数
    parser.add_argument(
        "--save-dir",
        type=str,
        default="attention_maps",
        help="图片保存的根目录",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="仅处理多少个 Batch 进行绘图",
    )

    # 添加模型定义参数 (来自 train.py)
    add_model_arguments(parser)
    # 添加数据加载参数
    AishellAsrDataModule.add_arguments(parser)

    return parser

# 全局字典用于存储捕获的权重
captured_weights = {}

def get_activation_hook(name):
    """
    创建一个 Hook 函数，用于在 Forward 时捕获输出
    """
    def hook(model, input, output):
        # output 是 attn_weights
        # Shape: (num_heads, batch_size, seq_len_tgt, seq_len_src)
        # 我们将其 detach 并转到 CPU
        captured_weights[name] = output.detach().cpu()
    return hook

def register_hooks(model):
    """
    遍历 Zipformer 模型结构，为所有 self_attn_weights 模块注册 Hook
    """
    print("正在注册 Attention Hooks...")
    
    # Zipformer 的结构通常是 model.encoder.encoders (ModuleList)
    # 其中每个元素可能是 DownsampledZipformer2Encoder 或 Zipformer2Encoder
    
    if not hasattr(model, 'encoder') or not hasattr(model.encoder, 'encoders'):
        logging.warning("未找到标准的 Zipformer encoder 结构，尝试递归搜索...")
        # 简单递归兜底
        for name, module in model.named_modules():
            if "self_attn_weights" in name:
                module.register_forward_hook(get_activation_hook(name))
        return

    encoder_stacks = model.encoder.encoders
    
    for stack_idx, encoder_module in enumerate(encoder_stacks):
        # 处理 DownsampledZipformer2Encoder 包装器
        if hasattr(encoder_module, 'encoder'):
            real_encoder = encoder_module.encoder
            prefix = f"Stack{stack_idx}_Downsampled"
        else:
            real_encoder = encoder_module
            prefix = f"Stack{stack_idx}"

        # 遍历 Stack 中的 Layers
        if hasattr(real_encoder, 'layers'):
            for layer_idx, layer in enumerate(real_encoder.layers):
                # 目标模块是 layer.self_attn_weights
                if hasattr(layer, 'self_attn_weights'):
                    hook_name = f"{prefix}_Layer{layer_idx}"
                    layer.self_attn_weights.register_forward_hook(get_activation_hook(hook_name))
                    print(f"  [Registered]: {hook_name}")



def plot_attention_weights(save_dir):
    """
    将捕获的权重绘制为热力图 (增强版)
    """
    if not captured_weights:
        logging.warning("没有捕获到任何 Attention 权重！")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    for name, weights in captured_weights.items():
        # weights shape: (num_heads, batch_size, T_tgt, T_src)
        # 取 Batch 中的第一个样本 -> (num_heads, T_tgt, T_src)
        sample_weights = weights[:, 0, :, :].float().numpy()
        
        # --- [DEBUG] 打印统计信息 ---
        # 这能帮你判断是数据全是0，还是单纯显示不出来
        max_val = np.max(sample_weights)
        min_val = np.min(sample_weights)
        mean_val = np.mean(sample_weights)
        print(f"Layer: {name}")
        print(f"  Shape: {sample_weights.shape}")
        print(f"  Stats -> Max: {max_val:.4f}, Min: {min_val:.4f}, Mean: {mean_val:.4f}")
        
        if max_val < 1e-5:
            logging.warning(f"  [Warning] {name} 的权重几乎全为0，可能模型未正确加载或输入异常。")
        # ---------------------------

        num_heads = sample_weights.shape[0]
        t_tgt = sample_weights.shape[1]
        t_src = sample_weights.shape[2]
        
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        # 增大图片尺寸，防止像素压缩导致看不清
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
        fig.suptitle(f"{name} (T_tgt:{t_tgt}, T_src:{t_src})\nMax:{max_val:.2f} Mean:{mean_val:.4f}", fontsize=14)
        
        axes_flat = axes.flatten() if num_heads > 1 else [axes]
        
        for head_idx in range(num_heads):
            ax = axes_flat[head_idx]
            attn_map = sample_weights[head_idx]
            
            # --- [核心修改] 使用 Log 变换增强对比度 ---
            # 原始概率往往差异极大，取 Log 可以让微小的关注点也显示出来
            # 加 1e-9 防止 log(0)
            # attn_map_log = np.log1p(attn_map * 100) # 这里的 *100 是为了拉开差距，可视效果更好
            
            # 也可以直接画原始图，但建议加上 vmin/vmax
            # im = ax.imshow(attn_map, cmap='viridis', aspect='auto', origin='upper', vmin=0, vmax=1.0)
            
            # 这里演示画 Log 变换后的图
            im = ax.imshow(attn_map, cmap='viridis', aspect='auto', origin='upper')
            
            ax.set_title(f"Head {head_idx}")
            if head_idx % cols == 0:
                ax.set_ylabel("Query (Tgt)")
            if head_idx >= (rows - 1) * cols:
                ax.set_xlabel("Key (Src)")
                
            # 添加颜色条以便观察数值范围
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        for i in range(num_heads, len(axes_flat)):
            axes_flat[i].axis('off')
            
        filename = os.path.join(save_dir, f"{name}.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"  [Saved]: {filename}")

def compute_features(wav_path, device):
    """读取音频并计算 Fbank 特征 (强制在 CPU 上进行以避免 cuFFT 错误)"""
    logging.info(f"Loading wave: {wav_path}")
    
    # 1. 加载音频 (默认在 CPU)
    waveform, sample_rate = torchaudio.load(wav_path)
    
    # 2. 重采样 (在 CPU 上进行)
    if sample_rate != 16000:
        logging.warning(f"Resampling from {sample_rate} to 16000 Hz")
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    
    # 【核心修改点】：不要在这里把 waveform 转到 device (GPU)
    # waveform = waveform.to(device)  <-- 删除这一行
    
    # 3. 计算 Fbank 特征 (在 CPU 上进行，避免调用 GPU 的 cuFFT)
    opts = {
        "num_mel_bins": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "dither": 0.0,
        "energy_floor": 0.0,
        "sample_frequency": 16000,
    }
    # (T, 80)
    features = kaldi.fbank(waveform, **opts)
    
    # 4. 计算完特征后，再移动到 GPU 传给模型
    features = features.unsqueeze(0).to(device) # (1, T, 80)
    
    # 获取长度 (也需要放在 GPU 上)
    features_lens = torch.tensor([features.shape[1]], device=device)
    
    return features, features_lens

@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.save_dir) / f"img-{timestamp}"
    setup_logger(f"{out_dir}/log-visualize")
    logging.info(f"Saving images to: {out_dir}")

    params = get_params()
    params.update(vars(args))
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

    lexicon = Lexicon(params.lang_dir)
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1

    # 1. 创建模型
    logging.info("Creating model...")
    model = get_model(params)

    # 2. 加载权重 (复用 decode.py 的逻辑)
    if not params.use_averaged_model:
        if params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"Averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[: params.avg + 1]
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(f"Averaging iteration checkpoints from {filename_start} to {filename_end}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            start = params.epoch - params.avg
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(f"Averaging epoch checkpoints from {start} to {params.epoch}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()

    # 3. 注册 Hook
    register_hooks(model)

    # 4. 准备数据
    logging.info("Preparing data...")
    try:
        feature, feature_lens = compute_features(args.wav_path, device)
    except Exception as e:
        logging.error(f"读取音频失败: {e}")
        return
    

    # 5. 运行推理
    logging.info("Running forward pass...")
    
    # 5.1 处理 Causal Padding (如果有)
    if params.causal:
        pad_len = 30
        feature_lens += pad_len
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, pad_len),
            value=torch.log(torch.tensor(1e-10)),
        )

    # 5.2 [关键步骤] 运行 Encoder Embedding (卷积下采样 + 维度投影)
    # Input: (N, T, 80) -> Output: (N, T_sub, encoder_dim)
    # 这步如果不做，维度不对，且缺少卷积提取的特征
    x, x_lens = model.encoder_embed(feature, feature_lens)

    # 5.3 [关键步骤] 生成 Mask 并调整维度
    src_key_padding_mask = make_pad_mask(x_lens)
    
    # Zipformer Encoder 需要 (T, N, C) 的输入形状
    # x: (N, T_sub, C) -> (T_sub, N, C)
    x = x.permute(1, 0, 2)

    # 5.4 运行 Encoder
    # 此时 x 的长度 T_sub 才是正确的时间步长
    _ = model.encoder(x, x_lens, src_key_padding_mask)
    # ================= 修改结束 =================
    
    logging.info("Generating images...")
    plot_attention_weights(out_dir)

    logging.info(f"Done! Check results in {out_dir}")

if __name__ == "__main__":
    main()