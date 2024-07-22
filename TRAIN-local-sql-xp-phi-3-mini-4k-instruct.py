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

# load tokens
import os

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
    pipeline,
)
from trl import SFTTrainer
from huggingface_hub import HfFolder
from dotenv import load_dotenv

load_dotenv()

# Load token from environment variable
hf_token = os.getenv("HF_TOKEN")

# logging into Hugging Face
# Save the token
HfFolder.save_token(hf_token)

# Preparing datasets

# DATASET_NAME is a string that specifies the name of the dataset to be used for fine-tuning.
DATASET_NAME = synthetic_text_to_sql_dataset_name = "gretelai/synthetic_text_to_sql"

# Load the dataset specified by DATASET_NAME using the load_dataset function.
dataset = load_dataset(DATASET_NAME, data_dir="./dataset/")


# Extract relevant fields
def extract_fields_synthetic(example):
    return {
        "instruction": example["sql_prompt"],
        "input": example["sql_context"],
        "output": example["sql"],
    }


synthetic_extracted_dataset = dataset.map(
    extract_fields_synthetic, remove_columns=dataset["train"].column_names
)

# Split and shuffle datasets

import random

synthetic_extracted_train_dataset = synthetic_extracted_dataset["train"]
synthetic_extracted_test_dataset = synthetic_extracted_dataset["test"]

# Shuffle the dataset
synthetic_extracted_dataset = synthetic_extracted_dataset.shuffle(
    seed=random.randint(10, 99)
)
synthetic_extracted_dataset = synthetic_extracted_dataset.shuffle(
    seed=random.randint(10, 99)
)

print(synthetic_extracted_train_dataset)
print(synthetic_extracted_test_dataset)

# attention and compute dtype

# 'torch.cuda.is_bf16_supported()' is a function that checks if BF16 is supported on the current GPU. BF16 is a data type that uses 16 bits, like float16, but allocates more bits to the exponent, which can result in higher precision.
# 'attn_implementation' is a variable that will hold the type of attention implementation to be used.

# if torch.cuda.is_bf16_supported():
#     compute_dtype = torch.bfloat16
# else:
#     compute_dtype = torch.float16

compute_dtype = torch.bfloat16
attn_implementation = "eager"
print(attn_implementation)
print(compute_dtype)

# Model info

# MODEL_ID is a string that specifies the identifier of the pre-trained model that will be fine-tuned.
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# NEW_MODEL_NAME is a string that specifies the name of the new model after fine-tuning.
NEW_MODEL_NAME = "sql-xp-phi-3-mini-4k"

# load tokenizr to prepare dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right"  # to prevent warnings

# Define methods for creating and formatting messages/prompts for datasets
# The prompt will contain our instructions, and the context will include the SQL context, such as a table creation SQL command


def create_message_column(row):
    """
    Create a message column for a dataset row.
    Args: row (dict): A dictionary containing 'instruction', 'input', and 'output' keys.
    Returns: dict: A dictionary with the key 'message' containing a list of messages for user and assistant.
    """
    message = []

    # Define the user message with prompt and context
    user = {
        "content": f"\n #prompt: {row['instruction']}\n #context: {row['input']}",
        "role": "user",
    }
    message.append(user)

    # Define the assistant's response
    assistant = {"content": f"{row['output']}", "role": "assistant"}
    message.append(assistant)

    # Return the constructed message
    return {"message": message}


def format_dataset_with_chat_template(row):
    """
    Format a dataset row using the chat template for tokenization.
    Args: row (dict): A dictionary containing the 'message' key.
    Returns:dict: A dictionary with the key 'text' containing the formatted text.
    """
    # Apply the chat template to the message and return the formatted text
    return {
        "text": tokenizer.apply_chat_template(
            row["message"], add_generation_prompt=False, tokenize=False
        )
    }


# Apply create_message_column function
synthetic_extracted_train_dataset = synthetic_extracted_train_dataset.map(
    create_message_column
)
synthetic_extracted_test_dataset = synthetic_extracted_test_dataset.map(
    create_message_column
)

# Format dataset using
synthetic_extracted_train_dataset = synthetic_extracted_train_dataset.map(
    format_dataset_with_chat_template
)
synthetic_extracted_test_dataset = synthetic_extracted_test_dataset.map(
    format_dataset_with_chat_template
)

# Output the results to verify
print(synthetic_extracted_train_dataset)
print(synthetic_extracted_test_dataset)

# select subsets of datasets
# 75:25 dataset ratio

synthetic_extracted_train_dataset = synthetic_extracted_train_dataset.select(
    range(10000)
)
synthetic_extracted_test_dataset = synthetic_extracted_test_dataset.select(range(3300))


# model configs for training

# 'hf_model_repo' is the identifier for the Hugging Face repository where you want to save the fine-tuned model.
hf_model_repo = "spectrewolf8/" + NEW_MODEL_NAME

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
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration for the model

# 'lora_r' or 'r' is the dimension of the LoRA attention.
# 'lora_alpha' is the alpha parameter for LoRA scaling.
# 'lora_dropout' is the dropout probability for LoRA layers.
# 'target_modules' is a list of the modules that should be targeted by LoRA.
# peft configuration for the model

lora_r = 16  # 16 default
peft_config = LoraConfig(
    lora_alpha=16,  # 16 default
    lora_dropout=0.05,  # 0.05 default
    r=lora_r,
    target_modules=[
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
)

# 'AutoTokenizer' is a class from the Hugging Face Transformers library that provides a tokenizer for a given pre-trained model.
# 'from_pretrained' is a method of the 'AutoTokenizer' class that loads a tokenizer from a pre-trained model.
# 'trust_remote_code=True' is a parameter that allows the execution of remote code when loading the tokenizer.
# 'add_eos_token=True' is a parameter that adds an end-of-sentence token to the tokenizer.
# 'use_fast=True' is a parameter that uses the fast version of the tokenizer, if available.
# 'tokenizer.pad_token = tokenizer.unk_token' sets the padding token of the tokenizer to be the same as the unknown token.
# 'tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)' sets the ID of the padding token to be the same as the ID of the padding token.
# 'tokenizer.padding_side = 'left'' sets the side where padding will be added to be the left side.
# 'BitsAndBytesConfig' is a class that provides a configuration for quantization.
# 'bnb_config' is a variable that holds the configuration for quantization.

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, add_eos_token=True, use_fast=True
)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "left"

# 'AutoModelForCausalLM' is a class from the Hugging Face Transformers library that provides a model for causal language modeling.
# 'from_pretrained' is a method of the 'AutoModelForCausalLM' class that loads a model from a pre-trained model.
# 'torch_dtype=compute_dtype' is a parameter that sets the data type of the model to be the same as 'compute_dtype'.
# 'quantization_config=bnb_config' is a parameter that sets the configuration for quantization to be 'bnb_config'.
# 'device_map=device_map' is a parameter that sets the device map of the model to be 'device_map'.
# 'attn_implementation=attn_implementation' is a parameter that sets the type of attention implementation to be 'attn_implementation'.
# 'model = prepare_model_for_kbit_training(model)' prepares 'model' for k-bit training and assigns the result back to 'model'.

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map=device_map,
    attn_implementation=attn_implementation,
)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# This block of code is used to initialize Weights & Biases (wandb) for experiment tracking.

# Retrieve the Weights & Biases API token from user secrets
wandb_token = os.getenv("WANDB_TOKEN")

# Import the wandb library for experiment tracking
import wandb

# Log in to Weights & Biases using the retrieved API token
wandb.login(key=wandb_token)

# Initialize a new Weights & Biases run for tracking the experiment
run = wandb.init(
    project="Training and tuning Phi-3-mini-4k-instruct for SQL | kaggle-sql-xp-phi-3-mini-4k-instruct.ipynb",
    job_type="training",  # Specify the type of job as training
    anonymous="allow",  # Allow anonymous logging if no user is logged in
)


## Training the model

# 'TrainingArguments' is a class from the Hugging Face Transformers library that provides hyperparameters for training.
# 'output_dir="./results"' sets the directory where the training results (like checkpoints and logs) will be saved.
# 'num_train_epochs=1' sets the number of times the entire training dataset will be passed through the model.
# 'per_device_train_batch_size=4' sets the batch size for training on each device (e.g., GPU).
# 'gradient_accumulation_steps=1' sets the number of steps to accumulate gradients before performing a backward/update pass.
# 'optim="paged_adamw_32bit"' specifies the optimizer to use; in this case, "paged_adamw_32bit" is used.
# 'save_steps=25' specifies the number of steps before saving a checkpoint.
# 'logging_steps=10' specifies the number of steps before logging training metrics.
# 'learning_rate=2e-4' sets the learning rate for the optimizer.
# 'weight_decay=0.001' applies weight decay (L2 regularization) to prevent overfitting.
# 'fp16=False' specifies whether to use 16-bit (half-precision) floating point.
# 'bf16=False' specifies whether to use bfloat16 precision (an alternative to fp16).
# 'max_grad_norm=0.3' clips the gradient norm to prevent the exploding gradient problem.
# 'max_steps=-1' specifies the total number of training steps; -1 means no limit.
# 'warmup_ratio=0.03' sets the proportion of training steps to perform learning rate warmup.
# 'group_by_length=True' groups sequences of similar lengths together for efficient training.
# 'lr_scheduler_type="constant"' specifies the type of learning rate scheduler; in this case, it uses a constant learning rate.
# 'report_to="wandb"' specifies the reporting tool to use for logging; in this case, Weights and Biases (wandb) is used.

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,  # 1 default
    per_device_train_batch_size=4,  # 4 default
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=5,  # 10 default but that's just for logging
    learning_rate=2e-4,  # 1.41e-5 default
    weight_decay=0.001,
    fp16=False,
    bf16=False,  # False default
    max_grad_norm=0.3,  # 0.3 default
    max_steps=-1,
    warmup_ratio=0.03,  # 0.03 default
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="wandb",
)

# 'SFTTrainer' is a class that provides a trainer for fine-tuning a model.
# 'trainer' is a variable that holds the trainer.
# 'model=model' is a parameter that sets the model to be trained to be 'model'.
# 'train_dataset=synthetic_extracted_train_dataset' is a parameter that sets the training dataset to be 'synthetic_extracted_train_dataset'.
# 'eval_dataset=synthetic_extracted_test_dataset' is a parameter that sets the evaluation dataset to be 'synthetic_extracted_test_dataset'.
# 'peft_config=peft_config' is a parameter that sets the configuration for the Lora layer to be 'peft_config'.
# 'dataset_text_field="text"' is a parameter that sets the field in the dataset that contains the text to be 'text'.
# 'max_seq_length=512' is a parameter that sets the maximum sequence length for the model to be 512.
# 'tokenizer=tokenizer' is a parameter that sets the tokenizer to be 'tokenizer'.
# 'args=args' is a parameter that sets the training arguments to be 'args'.
# This line of code is used to create a trainer for fine-tuning the model with the specified parameters.

trainer = SFTTrainer(
    model=model,
    train_dataset=synthetic_extracted_train_dataset,
    eval_dataset=synthetic_extracted_test_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,  # 512 default
    tokenizer=tokenizer,
    args=training_arguments,
)

# 'trainer.train()' is a method that starts the training of the model. It uses the training dataset, model, and training arguments that were specified when the trainer was created.

# train the model
trainer.train()

# 'trainer.save_model()' is a method that saves the trained model to the local file system. The model will be saved in the output directory that was specified in the training arguments.
# This block of code is used to train the model and then save the trained model to the local file system.

# save model locally
trainer.save_model()
tokenizer.save_pretrained("./results")

# Define the repository name on the Hugging Face Hub where the model, trainer, and tokenizer will be pushed.
hf_model_repo = "spectrewolf8/sql-xp-phi-3-mini-4k"

# Push the trainer to the Hugging Face Hub.
# This includes training arguments, optimizer states, and other relevant information.
trainer.push_to_hub(hf_model_repo)

# Push the model to the Hugging Face Hub.
# This saves the model weights and configuration to the specified repository.
trainer.model.push_to_hub(hf_model_repo)

# Push the tokenizer to the Hugging Face Hub.
# This saves the tokenizer configuration and vocab files to the specified repository.
tokenizer.push_to_hub(hf_model_repo)

# Finish the Weights & Biases run

# Finish the Weights & Biases (wandb) run.
# This finalizes the current experiment run, ensuring all data is uploaded and the run is properly closed.
wandb.finish()

# Set the 'use_cache' configuration option of the model to True.
# This enables caching of the computation results during inference, which can speed up the model's performance.
model.config.use_cache = True

# Set the model to evaluation mode.
# This changes the model's behavior to inference mode, disabling features like dropout that are only used during training.
model.eval()
