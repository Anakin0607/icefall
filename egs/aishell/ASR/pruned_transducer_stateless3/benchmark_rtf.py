#!/usr/bin/env python3
# benchmark_rtf_conformer.py

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Optional

import k2
import torch
import torch.nn as nn
from aishell import AIShell
from asr_datamodule import AsrDataModule
from lhotse.cut import Cut

# 引入解码算法
from beam_search import (
    fast_beam_search_one_best,
    greedy_search_batch,
    modified_beam_search,
)

# 引入训练代码中的辅助函数
from train import add_model_arguments, get_params, get_transducer_model

# Icefall 工具
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)

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
        single_cpu: 强制 batch_size=1，强制使用 CPU。
        batch: 使用指定 batch_size 和 device 测试吞吐量。
        """,
    )
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-batches", type=int, default=100)

    # --- 原有参数 ---
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--avg", type=int, default=15)
    parser.add_argument("--use-averaged-model", type=str2bool, default=True)
    parser.add_argument("--exp-dir", type=str, default="pruned_transducer_stateless3/exp")
    parser.add_argument("--lang-dir", type=str, default="data/lang_char")
    parser.add_argument("--decoding-method", type=str, default="greedy_search")
    
    # Beam Search 相关参数
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--beam", type=float, default=4)
    parser.add_argument("--max-contexts", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=1)
    parser.add_argument("--max-sym-per-frame", type=int, default=1)
    
    # 兼容性参数 (Shallow fusion 等，虽然RTF测试通常不用，但为了加载模型不报错保留)
    parser.add_argument("--use-shallow-fusion", type=str2bool, default=False)
    parser.add_argument("--lm-type", type=str, default="rnn")
    parser.add_argument("--lm-scale", type=float, default=0.3)
    parser.add_argument("--tokens-ngram", type=int, default=2)
    parser.add_argument("--ngram-lm-scale", type=float, default=0.01)
    parser.add_argument("--backoff-id", type=int, default=500)

    parser.add_argument(
        "--datatang-prob",
        type=float,
        default=0.0,
        help="""The probability to select a batch from the
        aidatatang_200zh dataset.
        If it is set to 0, you don't need to download the data
        for aidatatang_200zh.
        """,
    )

    add_model_arguments(parser)
    return parser

def run_forward_and_decode(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    decoding_graph: Optional[k2.Fsa] = None,
):
    device = next(model.parameters()).device
    feature = batch["inputs"].to(device)
    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    # 1. Encoder Forward
    encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)

    # 2. Search
    if params.decoding_method == "fast_beam_search":
        fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
    elif params.decoding_method == "modified_beam_search":
        modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
    else:
        # Default to greedy
        greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )

def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    
    # 【关键修改】强制开启日志输出，防止 setup_logger 失败导致无输出
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    params = get_params()
    params.update(vars(args))
    
    # 模式设置
    if params.benchmark_mode == "single_cpu":
        logging.info("Mode: Single CPU (Forcing batch_size=1, device=cpu)")
        # 强制单线程推理
        torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)
        logging.info("Running in Single-Threaded mode (simulating edge device).")

        args.batch_size = 1
        args.max_duration = 10 
        device = torch.device("cpu")
    else:
        logging.info(f"Mode: Batch Inference")
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    # 日志设置
    params.res_dir = params.exp_dir / "rtf_benchmark"
    params.res_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在
    setup_logger(f"{params.res_dir}/log-rtf")

    # 模型加载
    lexicon = Lexicon(params.lang_dir)
    params.blank_id = 0
    params.vocab_size = max(lexicon.tokens) + 1
    
    logging.info("Loading model...")
    model = get_transducer_model(params)
    
    # 检查点加载逻辑 (复用 decode.py)
    if params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[: params.avg + 1]
            model.load_state_dict(average_checkpoints_with_averaged_model(filenames[-1], filenames[0], device), strict=False)
        else:
            start = params.epoch - params.avg
            model.load_state_dict(average_checkpoints_with_averaged_model(f"{params.exp_dir}/epoch-{start}.pt", f"{params.exp_dir}/epoch-{params.epoch}.pt", device), strict=False)
    else:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)

    model.to(device)
    model.eval()

    # 图准备
    decoding_graph = None
    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)

    # 数据准备
    args.return_cuts = True
    asr_datamodule = AsrDataModule(args)
    aishell = AIShell(manifest_dir=args.manifest_dir)
    test_cuts = aishell.test_cuts()
    
    # 过滤短音频
    def remove_short_utt(c: Cut):
        return ((c.num_frames - 7) // 2 + 1) // 2 > 0
    test_cuts = test_cuts.filter(remove_short_utt)
    
    test_dl = asr_datamodule.test_dataloaders(test_cuts)

    # Benchmark 循环
    total_duration = 0.0
    total_time = 0.0
    count = 0
    
    logging.info(f"Start Benchmarking ({params.num_batches} batches)...")

    for i, batch in enumerate(test_dl):
        if params.num_batches > 0 and count >= params.num_batches:
            break
            
        try:
            batch_dur = sum(c.duration for c in batch["supervisions"]["cut"])
            
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            with torch.no_grad():
                run_forward_and_decode(params, model, batch, decoding_graph)
                
            if device.type == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            if i >= params.num_warmup:
                total_time += (t1 - t0)
                total_duration += batch_dur
                count += 1
                if i % 10 == 0:
                    logging.info(f"Batch {i}: RTF = {(t1-t0)/batch_dur:.4f}")
            else:
                logging.info(f"Warmup {i}")

        except Exception as e:
            logging.error(f"Error: {e}")

    if total_duration > 0:
        rtf = total_time / total_duration
        print("\n" + "="*40)
        print(f"Final RTF: {rtf:.5f}")
        print(f"Total Audio: {total_duration:.2f}s")
        print("="*40 + "\n")
        logging.info(f"Final RTF: {rtf:.5f}")

if __name__ == "__main__":
    main()