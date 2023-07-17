import socket
import torch
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from optimum.intel.openvino import OVConfig, OVQuantizer
import datasets
import tracemalloc
import nncf
import datetime
from pathlib import Path

print(socket.gethostname())
_original_print = print

# Quantize weights only
weight_quantization_config = {
    "algorithm": "quantization",
    "initializer":{
        "batchnorm_adaptation": {
            "num_bn_adaptation_samples": 0,
        },
        "range":{
            "num_init_samples": 4,
        }
    },
    "weights": {
        "mode": "asymmetric",
        "per_channel": True,
        "bits": 8,
        "target_scopes": [
            "{re}.*embedding.*",
            "{re}.*matmul.*",
            "{re}.*addmm.*",
            "{re}.*baddmm.*",
            "{re}.*linear*",
        ],
    },
    "activations": {
        "ignored_scopes": [
            "{re}.*__add___.*",
            "{re}.*__radd___.*",
            "{re}.*layer_norm_.*",
            "{re}.*__truediv__*",
            "{re}.*__mul___.*",
            "{re}.*__rmul___.*",
            "{re}.*tanh_.*",
            "{re}.*pow_.*",
            "{re}.*matmul.*",
            "{re}.*addmm.*",
            "{re}.*baddmm.*",
            "{re}.*linear*",
        ]
    },
}


def export_weight_quant_model(MODEL_ID):
    OUT_DIR = './weight_quantization/' + MODEL_ID.replace('/', '--')
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    def print(*args, **kwargs):
        _original_print(*args, **kwargs)
        with open(Path(OUT_DIR, 'log_export.log'), 'a', encoding='utf-8') as f:
            _original_print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), *args, **kwargs, file=f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_cache=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # return
    ov_config = OVConfig(compression=weight_quantization_config)
    ov_config.target_device = "TRIAL"
    ov_config.log_dir = OUT_DIR
    tokenizer.pad_token = tokenizer.eos_token

    tracemalloc.start()
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte") and type(model.transformer.wte) != torch.nn.Embedding:
        print(model.config_class)
        from nncf.torch import register_module
        register_module(ignored_algorithms=[])(type(model.transformer.wte))

    def preprocess_fn(examples, tokenizer):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=256)

    quantizer = OVQuantizer.from_pretrained(model)
    calibration_dataset = quantizer.get_calibration_dataset(
        "wikitext", 
        dataset_config_name="wikitext-103-v1", 
        num_samples=100, 
        dataset_split='train', 
        preprocess_function=partial(preprocess_fn, tokenizer=tokenizer)
    )
    quantizer.quantize(calibration_dataset=calibration_dataset, save_directory=OUT_DIR, 
                        weights_only=True, quantization_config=ov_config)
    tokenizer.save_pretrained(OUT_DIR)
    print("Memory usage (current, peak)", tracemalloc.get_traced_memory())
    tracemalloc.stop()


def main():
    for model_id in [
        'databricks/dolly-v2-3b',
        # 'EleutherAI/gpt-neox-20b',
        # "facebook/opt-350m",
        # 'mosaicml/mpt-7b-instruct',
        # 'mosaicml/mpt-7b-chat'
    ][::-1]:
        print('=' * 10, model_id, '=' * 10)
        try:
            export_weight_quant_model(model_id)
        except Exception as e:
            print(e)
            print('FAIL')
            raise

main()