# Cheatsheet for Benchmark

### setup bench environment
```bash
git clone https://github.com/vuiseng9/optimum-intel
cd optimum-intel && git checkout lm-ov-bench
pip install -e .[openvino,nncf,tests]
pip install tokenizers matplotlib pandas py-cpuinfo
```

### Export IR
For exporting FP32 IR, hack is required to disable nncf json validator
```bash
# comment out around line 50, NNCFConfig.validate(nncf_dict)
# in /.../miniconda3/envs/<env>/lib/python3.8/site-packages/nncf/config/config.py

model_id=gpt2
cd optimum-intel/examples/openvino/language-modeling/
python run_clm.py \
    --model_name_or_path $model_id \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --do_train \
    --per_device_train_batch_size 8 \
    --max_steps 1 \
    --output_dir ./${model_id}-fp32-ov-kv-cache \
    --overwrite_output_dir \
    --nncf_compression_config configs/dummy_fp32.json
```

### Toy Benchmark
```bash
python benchmark-text-generation.py \
    -m vuiseng9/ov-gpt2-fp32-no-cache \
    --disable_cache \
    --sweep_length 32 64 \
    --n_beam 1 \
    --n_warmup 2 \
    --n_sample 3 \
    --bench_label toyrun
```