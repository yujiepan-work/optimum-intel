import os
import sys
import time
import logging as log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import socket
import json
from cpuinfo import get_cpu_info
from argparse import ArgumentParser, SUPPRESS
from copy import deepcopy

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

from functools import partialmethod
from transformers import AutoTokenizer, pipeline, set_seed
from optimum.intel.openvino import OVModelForCausalLM
from collections import defaultdict

def create_prompt_of_length(ctx_len, tokenizer):
    SAMPLE_TEXT="A Hare was making fun of the Tortoise one day for being so slow. " + \
    "Do you ever get anywhere? he asked with a mocking laugh. Yes, replied the Tortoise, " + \
    "and I get there sooner than you think. I'll run you a race and prove it. " + \
    "The Hare was much amused at the idea of running a race with the Tortoise, " + \
    "but for the fun of the thing he agreed. So the Fox, who had consented to act as judge, " + \
    "marked the distance and started the runners off. The Hare was soon far out of sight, " + \
    "and to make the Tortoise feel very deeply how ridiculous it was for him to try a race with a Hare, " + \
    "he lay down beside the course to take a nap until the Tortoise should catch up." + \
    "The Tortoise meanwhile kept going slowly but steadily, and, after a time, " + \
    "passed the place where the Hare was sleeping. But the Hare slept on very peacefully; " + \
    "and when at last he did wake up, the Tortoise was near the goal. " + \
    "The Hare now ran his swiftest, but he could not overtake the Tortoise in time."
    
    sample_token_ids = tokenizer(SAMPLE_TEXT)['input_ids']
    if ctx_len > len(sample_token_ids):
        prompt_token_ids = (sample_token_ids * (len(sample_token_ids)//ctx_len + 1))[:ctx_len]
    else:
        prompt_token_ids = sample_token_ids[:ctx_len]
    return tokenizer.decode(prompt_token_ids)


def create_openvino_text_generation_pipeline(model_id, use_cache, device="CPU"):
    log.info("Create text generation pipeline with model_id: {} on device: {}".format(model_id, device))
    log.info("KV Cache bool: {}".format(use_cache))
    model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_cache, device=device, bench_mode=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return pipe, model, tokenizer


def boxplot_by_token(benchdict, args):
    axis_fontsize=10
    plot_width = args.max_new_tokens*0.4
    for i, (input_seqlen, raw_latency) in enumerate(benchdict.items()):
        token_latency = np.array(raw_latency)
        freq, gen_seqlen = token_latency.shape
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(plot_width, 10))
        ax=axes[0]
        ax.boxplot(token_latency, showfliers=False)
        ax.set_xlabel('N-th generated token', fontsize=axis_fontsize)
        ax.set_ylabel('Latency (sec.)', fontsize=axis_fontsize)
        title_str='Input Length: {} | Generated Length: {} | freq: {} | host: {} | label: {}'.format(
            input_seqlen, gen_seqlen, freq, args.host, args.bench_label)
        ax.set_title(title_str, fontsize=10)
        # ax.table(cellText=[pd.Series(np.around(arr.mean(axis=0),4)).astype(str).tolist()],
        #          rowLabels=['mean'],
        #          colLabels=list(range(1, gen_seqlen+1)),
        #          loc='bottom',
        #          zorder=10)
        
        ax=axes[1]
        mean_vec = np.around(token_latency.mean(axis=0),4)
        ax.plot(range(1, gen_seqlen+1), mean_vec,  marker='.', linestyle='--', ms=12, label=f"{input_seqlen}", color=colors[i])
        # for i in range(1, gen_seqlen+1):
            # ax.annotate(mean_vec[i-1], (i, mean_vec[i-1]*1.05))
        ax.set_xticks(range(1, gen_seqlen+1))
        ax.set_xlabel('N-th generated token', fontsize=axis_fontsize)
        ax.set_ylabel('Mean Latency (sec.)', fontsize=axis_fontsize)
        ax.set_title(title_str, fontsize=10)
        ax.grid(linestyle='-.', linewidth=0.5)
        # min_vec = np.around(arr.min(axis=0),4)
        # max_vec = np.around(arr.max(axis=0),4)
        # ax.fill_between(range(1, gen_seqlen+1), min_vec, max_vec)
        ax.legend()
        filename='ctx{}_gen{}.png'.format(str(input_seqlen).zfill(4), str(gen_seqlen).zfill(4))
        plt.savefig(os.path.join(args.bench_outdir, filename))

    # all input length in one plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(plot_width, 5))
    for input_seqlen, raw_latency in benchdict.items():
        token_latency = np.array(raw_latency)
        freq, gen_seqlen = token_latency.shape
        
        mean_vec = np.around(token_latency.mean(axis=0),4)
        ax.plot(range(1, gen_seqlen+1), mean_vec,  marker='.', linestyle='--', ms=12, label=f"{input_seqlen}")
        # for i in range(1, gen_seqlen+1):
            # ax.annotate(mean_vec[i-1], (i, mean_vec[i-1]*1.05))
        ax.set_xticks(range(1, gen_seqlen+1))
        ax.set_xlabel('N-th generated token', fontsize=axis_fontsize)
        ax.set_ylabel('Mean Latency (sec.)', fontsize=axis_fontsize)
        title_str='N-th token latency over Input Length: {} \nGenerated Length: {} | freq: {} | host: {} | label: {}\n model: {}'.format(
            str(args.sweep_length), gen_seqlen, freq, args.host, args.bench_label, args.model)
        ax.set_title(title_str, fontsize=10)
        ax.grid(linestyle='-.', linewidth=0.5)
    ax.legend()
    filename='per_token_latency_by_prompt_length.png'
    plt.savefig(os.path.join(args.bench_outdir, filename))


def barplot_first_token_latency(benchdict, args):
    first_token_serieslist = []
    for input_seqlen, raw_latency in benchdict.items():
        first_token_latency = np.array(raw_latency)[:,:1].reshape(-1)
        first_token_serieslist.append(pd.Series(first_token_latency).describe())

    first_token_latency_df = pd.DataFrame(first_token_serieslist, index=benchdict.keys())
    # ax  = first_token_latency_df.plot.barh(y=["mean", "50%"])
    ax  = first_token_latency_df.plot.barh(y=["mean"], title=f"First token latency, by Prompt Length.\n model: {args.model}")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3fs', padding=5)
        ax.set_xlabel('latency (sec.)')
        ax.set_ylabel('Input Length')
    filename='first_token_latency.png'
    plt.savefig(os.path.join(args.bench_outdir, filename))


def barplot_beyond_token1_latency(benchdict, args):
    beyond_token1_serieslist = []
    for input_seqlen, raw_latency in benchdict.items():
        beyond_token1_latency = np.array(raw_latency)[:,1:].mean(axis=0) # mean over sampling loop
        beyond_token1_serieslist.append(pd.Series(beyond_token1_latency).describe())

    beyond_token1_latency_df = pd.DataFrame(beyond_token1_serieslist, index=benchdict.keys())
    ax = beyond_token1_latency_df.plot.barh(
        y=["max", "mean", "min"], 
        title=f"Latency of 2nd token and beyond, by Prompt Length.\n Generated Length: {args.max_new_tokens}, model: {args.model}")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3fs', padding=5)
        ax.set_xlabel('latency (sec.)')
        ax.set_ylabel('Input Length')
    filename='beyond_token1_latency.png'
    plt.savefig(os.path.join(args.bench_outdir, filename))


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. HuggingFace Model Id or local path, model must contain OV IR",
                      required=True, type=str)
    args.add_argument('--disable_cache', action='store_true', help='disable cache for computed KV tokens')
    args.add_argument("--max_new_tokens", help="Optional. Maximum number of New (generated) tokens in generation",
                      default=50, required=False, type=int)
    parser.add_argument('-l','--sweep_length', nargs='+', help='<Required> sweep space of input prompt length', required=True, type=int)
    args.add_argument("--n_sample", help="Optional. #sample of latency collection, default to 2",
                      default=2, required=False, type=int)
    args.add_argument("--n_warmup", help="Optional. warm up #loop for generation pipeline, default to 1",
                      default=1, required=False, type=int)
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on. "
                           "Default value is CPU",
                      default="CPU", type=str)
    args.add_argument("--bench_label", help="Required with --bench. benchmark identifier", type=str)
    args.add_argument("--bench_outdir", help="benchmark dump output dir, local dir with the name of label will be created if not provided", required=False, type=str)
    return parser

def main():
    args = build_argparser().parse_args()

    if args.bench_label is None:
        raise ValueError("Pls specify --bench_label <label> when running with --bench")
    
    if args.bench_outdir is None:
        args.bench_outdir = os.path.join("./", args.bench_label)


    os.makedirs(args.bench_outdir, exist_ok=True)
    args.host = socket.gethostname()
    args.system = get_cpu_info()['brand_raw']
    
    set_seed(42)
    model_id=args.model # "vuiseng9/ov-gpt2-fp32-no-cache"
    pipe, model, tokenizer = create_openvino_text_generation_pipeline(
        model_id, use_cache=(not args.disable_cache), device=args.device)

    # only create output when pipeline is created successfully, otherwise we need to delete unused folders
    with open(os.path.join(args.bench_outdir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=True)

    SWEEP_PROMPT_LENGTH = args.sweep_length
    SAMPING_LOOP = args.n_sample
    WARMUP_LOOP = args.n_warmup
    GEN_MAXLEN = args.max_new_tokens

    sweep_bench_per_token = defaultdict(list)
    sweep_bench_e2e = defaultdict(list)

    for ctx_len in SWEEP_PROMPT_LENGTH:
        log.info('-'*100)
        log.info(f"Benchmarking input prompt of length {ctx_len} ...")
        prompt = create_prompt_of_length(ctx_len, tokenizer)
        for _ in range(WARMUP_LOOP):
            _gen_output = pipe(prompt, max_new_tokens=GEN_MAXLEN, num_return_sequences=1)
        
        log.info(f"{WARMUP_LOOP} warm-up loop complete. Collecting {SAMPING_LOOP} samples ...")
        for loop_i in range(SAMPING_LOOP):
            model.reset_benchdata()
            e2e_start = time.perf_counter()
            _ = pipe(prompt, max_new_tokens=GEN_MAXLEN, num_return_sequences=1)
            e2e_end = time.perf_counter()

            # following is to ensure that we get equal length of all 
            while len(model.token_latency) != GEN_MAXLEN:
                log.warn("Generated #token != required #token")
                model.reset_benchdata()
                e2e_start = time.perf_counter()
                gen_output = pipe(prompt, max_new_tokens=GEN_MAXLEN, num_return_sequences=1)
                e2e_end = time.perf_counter()

            sweep_bench_e2e[ctx_len].append(e2e_end - e2e_start)
            log.debug(len(model.token_latency))
            sweep_bench_per_token[ctx_len].append(deepcopy(model.token_latency))
    
    import torch
    def pickle_obj(obj, label):
        torch.save(obj, os.path.join(args.bench_outdir, f"{label}.pth"))
    
    pickle_obj(sweep_bench_e2e, "sweep_bench_e2e")
    pickle_obj(sweep_bench_per_token, "sweep_bench_per_token")
    
    log.info('-'*100)
    log.info("Benchmark data collection complete and saved. Generating plots...")
    boxplot_by_token(sweep_bench_per_token, args)
    barplot_first_token_latency(sweep_bench_per_token, args)
    barplot_beyond_token1_latency(sweep_bench_per_token, args)
    


if __name__ == '__main__':
    sys.exit(main() or 0)
