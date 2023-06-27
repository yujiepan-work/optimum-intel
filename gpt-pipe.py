from functools import partialmethod
from transformers import AutoTokenizer, pipeline, set_seed, AutoModelForCausalLM
from optimum.intel.openvino import OVModelForCausalLM

# Pipeline with Native HF model 
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1)
print(output)

# Alternative Pipeline with Native HF model 
model_dir="/data1/vchua/run/hf-model/gpt2"
# model = AutoModelForCausalLM.from_pretrained(model_dir) # unused
tokenizer = AutoTokenizer.from_pretrained(model_dir)
generator_pipe = pipeline('text-generation', model=model_dir, tokenizer=tokenizer)
output = generator_pipe("I love the Avengers", max_length=30, num_return_sequences=1)
print(output)

# Pipeline with OpenVINO IR (no cache)
model_id="vuiseng9/ov-gpt2-fp32-no-cache"
model = OVModelForCausalLM.from_pretrained(model_id, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator_pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
output = generator_pipe("It's a beautiful day ...", max_length=30, num_return_sequences=1)
print(output[0]['generated_text'])

# Pipeline with OpenVINO IR (KV Cache)
model_id="vuiseng9/ov-gpt2-fp32-kv-cache"
model = OVModelForCausalLM.from_pretrained(model_id, use_cache=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator_pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
output = generator_pipe("It's a beautiful day ...", max_length=30, num_return_sequences=1)
print(output[0]['generated_text'])

print("- end of generation -")
