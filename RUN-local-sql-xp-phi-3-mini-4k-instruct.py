# Import necessary libraries
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

# Set the seed for the random number generator to ensure reproducibility
set_seed(1234)

# Define the repository name for the Hugging Face model
# 'hf_model_repo' is a variable that holds the repository name for the Hugging Face model
# 'username/modelname' is the repository name, where 'username' is the username of the repository owner
# and 'modelname' is the name of the model
hf_model_repo = "spectrewolf8/sql-xp-phi-3-mini-4k"

# Retrieve the device mapping and computation data type
# 'device_map' is a variable that holds the mapping of the devices that are used for computation
# 'compute_dtype' is a variable that holds the data type that is used for computation

# device_map = {"": 0}
# compute_dtype = torch.bfloat16 or torch.float16
device_map = {"": 0}
compute_dtype = torch.bfloat16

# Load a pre-trained tokenizer from the Hugging Face Model Hub
# 'tokenizer' is the variable that holds the tokenizer
# 'trust_remote_code=True' allows the execution of code from the model file
tokenizer = AutoTokenizer.from_pretrained(hf_model_repo, trust_remote_code=True)

# Load a pre-trained model for causal language modeling from the Hugging Face Model Hub
# 'model' is the variable that holds the model
# 'trust_remote_code=True' allows the execution of code from the model file
# 'torch_dtype=compute_dtype' sets the data type for the PyTorch tensors
# 'device_map=device_map' sets the device mapping
model = AutoModelForCausalLM.from_pretrained(
    hf_model_repo,
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    device_map=device_map,
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define the context and input phrase
context_phrase = """
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE projects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    project_name VARCHAR(100) NOT NULL,
    description TEXT,
    start_date DATE NOT NULL,
    end_date DATE,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE tasks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('pending', 'in_progress', 'completed')),
    priority INT CHECK (priority BETWEEN 1 AND 5),
    project_id INT,
    assigned_to INT,
    due_date DATE,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (assigned_to) REFERENCES users(id)
);

CREATE TABLE comments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id INT,
    user_id INT,
    comment_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

input_phrase = """
Update the status of tasks to 'completed' for all tasks that have passed their due date. Also, update the end date of the corresponding projects to the current date if all tasks in the project are completed.
"""

# Apply the chat template to create the prompt
prompt = pipe.tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": f"\n #prompt: {input_phrase}\n #context: {context_phrase}",
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
)

# Generate SQL query
outputs = pipe(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    num_beams=1,
    temperature=0.3,
    top_k=50,
    top_p=0.95,
    max_time=180,
)

# Print the result
generated_text = outputs[0]["generated_text"][len(prompt) :].strip()
print(f"Generated SQL Query:\n{generated_text}")
