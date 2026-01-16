#!/usr/bin/env python3
# benchmark_rtf.py

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Optional

import k2
import torch
import torch.nn as nn
from asr_datamodule import AishellAsrDataModule
from lhotse.cut import Cut

# 复用 icefall 的工具
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
from icefall.lexicon import Lexicon
from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler

# 引入训练代码中的辅助函数
from train_kvpooling import add_model_arguments, get_model, get_params

# 引入解码搜索算法
from beam_search import (
    beam_search,
    fast_beam_search_one_best,
    fast_beam_search_nbest_oracle,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)

LOG_EPS = math.log(1e-10)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- RTF 测试专用参数 ---
    parser.add_argument(
        "--benchmark-mode",
        type=str,
        default="single_cpu",
        choices=["single_cpu", "batch"],
        help="""
        single_cpu: 强制 batch_size=1，强制使用 CPU，模拟单条音频推理。
        batch: 使用指定 batch_size 和 device (CPU/GPU) 进行推理，测试吞吐量。
        """,
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="预热的 batch 数量，不计入总时间。",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="用于测试的 batch 数量。设为 -1 则跑完整个测试集。",
    )
    # -----------------------

    # 以下参数复用自 decode.py，为了正确加载模型
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--avg", type=int, default=15)
    parser.add_argument("--use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--exp-dir", type=str, default="zipformer/exp")
    parser.add_argument("--lang-dir", type=Path, default="data/lang_char")
    parser.add_argument("--decoding-method", type=str, default="greedy_search")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--beam", type=float, default=20.0)
    parser.add_argument("--ngram-lm-scale", type=float, default=0.01)
    parser.add_argument("--ilme-scale", type=float, default=0.2)
    parser.add_argument("--max-contexts", type=int, default=8)
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--context-size", type=int, default=2)
    parser.add_argument("--max-sym-per-frame", type=int, default=1)
    parser.add_argument("--num-paths", type=int, default=200)
    parser.add_argument("--nbest-scale", type=float, default=0.5)
    parser.add_argument("--blank-penalty", type=float, default=0.0)

    add_model_arguments(parser)
    return parser

def run_forward_and_decode(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    decoding_graph: Optional[k2.Fsa] = None,
    lexicon: Lexicon = None,
    graph_compiler: CharCtcTrainingGraphCompiler = None,
):
    """
    执行一次完整的推理过程（Encoder + Search），不返回具体文本，只运行过程。
    代码逻辑直接复用 decode_one_batch 的核心部分。
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    if params.causal:
        pad_len = 30
        feature_lens += pad_len
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, pad_len),
            value=LOG_EPS,
        )

    # 1. Encoder Forward
    x, x_lens = model.encoder_embed(feature, feature_lens)
    src_key_padding_mask = make_pad_mask(x_lens)
    x = x.permute(1, 0, 2)
    encoder_out, encoder_out_lens = model.encoder(x, x_lens, src_key_padding_mask)
    encoder_out = encoder_out.permute(1, 0, 2)

    # 2. Decoding Search (模拟真实解码开销)
    if params.decoding_method == "fast_beam_search":
        fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            blank_penalty=params.blank_penalty,
        )
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            blank_penalty=params.blank_penalty,
        )
    elif params.decoding_method == "modified_beam_search":
        modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            blank_penalty=params.blank_penalty,
            beam=params.beam_size,
        )
    else:
        # 对于其他更复杂的解码方法，为简单起见，这里默认回退到 batch greedy search
        # 或者你可以根据需要把 decode.py 里的完整逻辑拷过来
        greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            blank_penalty=params.blank_penalty,
        )
    return

def main():
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # --- 模式设置 ---
    if params.benchmark_mode == "single_cpu":
        logging.info("Mode: Single CPU (Forcing batch_size=1, device=cpu)")
        params.batch_size = 1  # 强制 Batch Size 为 1
        # 注意：DataModule 里的 dataloader 参数通常在 argparse 解析时已经设定
        # 我们需要在初始化 DataModule 之前覆盖它，或者在 DataModule 初始化后 hack
        args.batch_size = 1 # 覆盖 args，传递给 DataModule
        args.max_duration = 10 # 设置一个很大的值，确保不会因为 duration 限制导致 batch 变大，反正 batch_size=1 限制了
        device = torch.device("cpu")

        # 强制单线程推理
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        logging.info("Running in Single-Threaded mode (simulating edge device).")
    else:
        # logging.info(f"Mode: Batch Inference (Batch Size: {params.batch_size})")
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
    
    logging.info(f"Active Device: {device}")

    # --- 模型加载 (复用逻辑) ---
    params.res_dir = params.exp_dir / "rtf_benchmark"
    setup_logger(f"{params.res_dir}/log-rtf")
    logging.info("RTF benchmark started!")

    lexicon = Lexicon(params.lang_dir)
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1
    
    # 图构建（虽然不一定用，但为了兼容性加载）
    graph_compiler = CharCtcTrainingGraphCompiler(lexicon=lexicon, device=device)
    
    model = get_model(params)
    
    # 模型平均加载逻辑 (简化版，直接从 decode.py 逻辑复制)
    if params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[: params.avg + 1]
            filename_start, filename_end = filenames[-1], filenames[0]
            model.load_state_dict(average_checkpoints_with_averaged_model(filename_start, filename_end, device))
        else:
            start = params.epoch - params.avg
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            model.load_state_dict(average_checkpoints_with_averaged_model(filename_start, filename_end, device))
    else:
         load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)

    model.to(device)
    model.eval()

    # 加载解码图
    decoding_graph = None
    if "fast_beam_search" in params.decoding_method:
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)

    # --- 数据加载 ---
    args.return_cuts = True # 必须开启以获取音频时长
    aishell = AishellAsrDataModule(args)
    test_cuts = aishell.test_cuts()
    
    # 过滤短音频
    def remove_short_utt(c: Cut):
        T = ((c.num_frames - 7) // 2 + 1) // 2
        return T > 0
    test_cuts = test_cuts.filter(remove_short_utt)
    
    # 获取 DataLoader
    dl = aishell.test_dataloaders(test_cuts)

    # --- Benchmark 主循环 ---
    total_duration_sec = 0.0
    total_process_time_sec = 0.0
    processed_batches = 0
    
    logging.info(f"Start benchmarking... (Warmup: {params.num_warmup} batches)")

    for batch_idx, batch in enumerate(dl):
        # 0. 退出条件
        if params.num_batches > 0 and processed_batches >= params.num_batches:
            break

        # 1. 获取当前 Batch 的音频总时长 (秒)
        # supervisions['cut'] 包含 lhotse cut 对象，里面有 duration
        batch_duration = sum(c.duration for c in batch["supervisions"]["cut"])

        # 2. 计时推理
        try:
            # 同步 CUDA (Start)
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()

            # --- 运行推理 ---
            with torch.no_grad():
                run_forward_and_decode(
                    params=params,
                    model=model,
                    batch=batch,
                    decoding_graph=decoding_graph,
                    lexicon=lexicon,
                    graph_compiler=graph_compiler
                )
            
            # 同步 CUDA (End)
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            # 3. 统计 (跳过 Warmup)
            if True :# batch_idx >= params.num_warmup:
                total_process_time_sec += elapsed
                total_duration_sec += batch_duration
                processed_batches += 1
                
                # 实时打印每一步的 RTF
                current_rtf = elapsed / batch_duration
                if batch_idx % 10 == 0:
                    logging.info(f"Batch {batch_idx}: Duration={batch_duration:.2f}s, Time={elapsed:.4f}s, RTF={current_rtf:.4f}")

            else:
                logging.info(f"Warmup Batch {batch_idx} finished.")

        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {e}")
            continue

    # --- 最终结果报告 ---
    if total_duration_sec > 0:
        avg_rtf = total_process_time_sec / total_duration_sec
        
        print("\n" + "="*50)
        print(f"Benchmark Result ({params.benchmark_mode})")
        print("="*50)
        print(f"Device          : {device}")
        print(f"Decoding Method : {params.decoding_method}")
        print(f"Batches Tested  : {processed_batches}")
        print(f"Total Audio     : {total_duration_sec:.2f} seconds")
        print(f"Total Time      : {total_process_time_sec:.2f} seconds")
        print("-" * 50)
        print(f"Final RTF       : {avg_rtf:.5f}")
        print("="*50 + "\n")
        
        logging.info(f"Final RTF: {avg_rtf:.5f}")
    else:
        logging.error("No data processed!")

if __name__ == "__main__":
    main()