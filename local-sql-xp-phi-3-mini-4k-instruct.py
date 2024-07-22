# !pip install flash_attn==2.5.8
# !pip install torch==2.3.1
# !pip install accelerate==0.31.0
# !pip install transformers==4.41.2
# !pip install datasets
# !pip install transformers
# !pip install trl
# !pip install peft 
# !pip install auto-gptq 
# !pip install optimum
# !pip install xformers
# !pip install huggingface_hub
# !pip install git+https://github.com/microsoft/LoRA
    
# #bits and bytes with cuda
# !pip install bitsandbytes-cuda110 bitsandbytes


#load tokens
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")

#logging into Hugging Face
!huggingface-cli login --token $hf_token

# impoting classes
from random import randrange

import torch
from datasets import load_dataset

from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)
from trl import SFTTrainer

# Preparing datasets

# DATASET_NAME is a string that specifies the name of the dataset to be used for fine-tuning.
DATASET_NAME = synthetic_text_to_sql_dataset_name = "gretelai/synthetic_text_to_sql"

# Load the dataset specified by DATASET_NAME using the load_dataset function.
dataset = load_dataset(DATASET_NAME)

dataset

# Extract relevant fields

# old
# def extract_fields_synthetic(example):
#     return {
#         "question": example["sql_prompt"],
#         "context": example["sql_context"],
#         "sql": example["sql"]
#     }

# new
def extract_fields_synthetic(example):
    return {
        "instruction": example["sql_prompt"],
        "input": example["sql_context"],
        "output": example["sql"]
    }
synthetic_extracted_dataset = dataset.map(extract_fields_synthetic, remove_columns=dataset['train'].column_names)


# Split and shuffle datasets

import random 

synthetic_extracted_train_dataset = synthetic_extracted_dataset["train"]
synthetic_extracted_test_dataset = synthetic_extracted_dataset["test"]

# Shuffle the dataset
synthetic_extracted_dataset = synthetic_extracted_dataset.shuffle(seed=random.randint(10,99))
synthetic_extracted_dataset = synthetic_extracted_dataset.shuffle(seed=random.randint(10,99))

print(synthetic_extracted_train_dataset)
print(synthetic_extracted_test_dataset)

# 'torch.cuda.is_bf16_supported()' is a function that checks if BF16 is supported on the current GPU. BF16 is a data type that uses 16 bits, like float16, but allocates more bits to the exponent, which can result in higher precision.
# 'attn_implementation' is a variable that will hold the type of attention implementation to be used.

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
else:
  compute_dtype = torch.float16

attn_implementation = 'eager'
print(attn_implementation)
print(compute_dtype)

#load tokenizr to prepare dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = 'right' # to prevent warnings

#Define message/prompt creation and formatting methods for the datasets
# #prompt will have our prompt/instruction
# #context will have out SQL context i.e table creation sql command

def create_message_column(row):
    message = []
    user = {
        "content": f"\n #prompt: {row['instruction']}\n #context: {row['input']}",
        "role": "user"
    }
    message.append(user)
    assistant = {
        "content": f"{row['output']}",
        "role": "assistant"
    }
    message.append(assistant)
    return {"message": message}

def format_dataset_with_chat_template(row):
    return {"text": tokenizer.apply_chat_template(row["message"], add_generation_prompt=False, tokenize=False)}

# Apply create_message_column function
synthetic_extracted_train_dataset = synthetic_extracted_train_dataset.map(create_message_column)
synthetic_extracted_test_dataset = synthetic_extracted_test_dataset.map(create_message_column)

# Format dataset using 
synthetic_extracted_train_dataset = synthetic_extracted_train_dataset.map(format_dataset_with_chat_template)
synthetic_extracted_test_dataset = synthetic_extracted_test_dataset.map(format_dataset_with_chat_template)

# Output the results to verify
print(synthetic_extracted_train_dataset)
print(synthetic_extracted_test_dataset)

#select subsets of datasets
# 75:25 dataset ratio

synthetic_extracted_train_dataset = synthetic_extracted_train_dataset.select(range(1000))
synthetic_extracted_test_dataset = synthetic_extracted_test_dataset.select(range(330))

# MODEL_ID is a string that specifies the identifier of the pre-trained model that will be fine-tuned. 
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# NEW_MODEL_NAME is a string that specifies the name of the new model after fine-tuning.
NEW_MODEL_NAME = "sql-xp-phi-3-mini-4k"

# 'hf_model_repo' is the identifier for the Hugging Face repository where you want to save the fine-tuned model.
hf_model_repo="spectrewolf8/"+NEW_MODEL_NAME

# Load Model on GPU 
# 'device_map' is set to {"": 0}, which means that the entire model will be loaded on GPU 0.
device_map = {"": 0}


# Bits and Bytes configuration for the model

# 'load_in_4bit' is a boolean that control if 4bit quantization should be loaded. In this case, it is set to True
# 'bnb_4bit_compute_dtype' is the data type that should be used for computations with the 4-bit base model. In this case, it is set to 'bfloat16'.
# 'bnb_4bit_quant_type' is the type of quantization that should be used for the 4-bit base model. In this case, it is set to 'nf4'.
# 'bnb_4bit_use_double_quant' is a boolean that controls whether nested quantization should be used for the 4-bit base model.

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
)

# LoRA configuration for the model

# 'lora_r' is the dimension of the LoRA attention.
# 'lora_alpha' is the alpha parameter for LoRA scaling.
# 'lora_dropout' is the dropout probability for LoRA layers.
# 'target_modules' is a list of the modules that should be targeted by LoRA.

lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

# peft configuration for the model
peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r = lora_r,
    target_modules=target_modules
)

# 'set_seed(1234)' sets the random seed for reproducibility.
set_seed(1234)

# username is a string that specifies the GitHub username of the person who is fine-tuning the model.
# license is a string that specifies the license under which the model is distributed. In this case, it's Apache License 2.0.

username = "spectrewolf8"
license = "apache-2.0"

# MAX_SEQ_LENGTH is an integer that specifies the maximum length of the sequences that the model will handle.
# num_train_epochs is an integer that specifies the number of times the training process will go through the entire dataset.
# learning_rate is a float that specifies the learning rate to be used during training.
# per_device_train_batch_size is an integer that specifies the number of samples to work through before updating the internal model parameters.
# gradient_accumulation_steps is an integer that specifies the number of steps to accumulate gradients before performing a backward/update pass.

MAX_SEQ_LENGTH = 2048
num_train_epochs = 1
learning_rate = 1.41e-5
per_device_train_batch_size = 4
gradient_accumulation_steps = 1
