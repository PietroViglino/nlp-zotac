import os, torch, logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Dataset
data_name = "Nan-Do/code-search-net-python"
training_data = load_dataset(data_name, split="train")
# Model and tokenizer names
base_model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"
refined_model = "llama-2-7b-python"
# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token='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')
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right" 
# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1