# kaggle-sql-xp-phi-3-mini-4k-instruct

> [!NOTE]
> library_name: transformers
> <br/>
> tags: [NLP, GPT-3, Chatbot, Instruction Tuning, SQL, SQL-Generation]

# Model Card for sql-xp-phi-3-mini

<!-- Phi-3 Mini is a transformer-based language model optimized for understanding and generating responses based on instructional input. -->



## Model Details

### Model Description

This model card describes Phi-3 Mini, a smaller variant of the Phi-3 series, designed to handle instructions with a 4k token context length. It is specifically fine-tuned to follow instructional prompts effectively, making it suitable for applications requiring interactive and responsive dialogue systems.

- **Developed by:** [spectrewolf8]
- **Model type:** Transformer-based Language Model
- **Language(s) (NLP):** English (and SQL)
- **License:** MIT
- **Finetuned from model [optional]:** Phi-3-mini-instruct-4k base model

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/Spectrewolf8/kaggle-sql-xp-phi-3-mini-4k-instruct

## Uses

### Direct Use

Phi-3 Mini can be used to translate natural language instructions into SQL queries, making it a powerful tool for database querying and management. Users can input descriptive text, and the model will generate the corresponding SQL commands.

### Downstream Use [optional]

This model can be integrated into applications such as chatbots or virtual assistants that interact with databases. It can also be used in tools designed for automatic query generation based on user-friendly descriptions.

### Out-of-Scope Use

Phi-3 Mini is not suitable for tasks requiring non-SQL-related language understanding or generation. It should not be used for generating queries in languages other than SQL or for other domains outside database querying.

### Bias, Risks, and Limitations

Phi-3 Mini, like other language models, may have limitations in understanding complex or ambiguous instructions. The SQL queries generated might need manual review to ensure accuracy and appropriateness.

### Recommendations

Users should verify the generated SQL queries for correctness and security, especially when using them in production environments. Implementing additional layers of validation and testing can help mitigate risks associated with incorrect SQL generation.


## How to Get Started with the Model

To get started with Phi-3 Mini for SQL generation, follow the code snippet below:

```
# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

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
device_map, compute_dtype

# Load a pre-trained tokenizer from the Hugging Face Model Hub
# 'tokenizer' is the variable that holds the tokenizer
# 'trust_remote_code=True' allows the execution of code from the model file
tokenizer = AutoTokenizer.from_pretrained(hf_model_repo, trust_remote_code=True)

# Load a pre-trained model for causal language modeling from the Hugging Face Model Hub
# 'model' is the variable that holds the model
# 'trust_remote_code=True' allows the execution of code from the model file
# 'torch_dtype=compute_dtype' sets the data type for the PyTorch tensors
# 'device_map=device_map' sets the device mapping
model = AutoModelForCausalLM.from_pretrained(hf_model_repo, trust_remote_code=True, torch_dtype=compute_dtype, device_map=device_map)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define the input phrase which represents the user's request or query.
input_phrase = """
insert 5 values
"""

# Define the context phrase which provides the SQL table schema relevant to the input phrase.
context_phrase = """
CREATE TABLE tasks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    task_name VARCHAR(100) NOT NULL,
    userid INT NOT NULL,
    date DATE NOT NULL,
    FOREIGN KEY (userid) REFERENCES users(id)
);
"""

# Create a prompt by applying a chat template to the input and context phrases using the tokenizer.
# The 'apply_chat_template' method formats the input as a chat message, making it suitable for text generation.
# 'tokenize=False' indicates that the input should not be tokenized yet.
# 'add_generation_prompt=True' adds a prompt for text generation.
prompt = pipe.tokenizer.apply_chat_template(
    [{"role": "user", "content": f"\n #prompt: {input_phrase}\n #context: {context_phrase}"}],
    tokenize=False,
    add_generation_prompt=True
)

# Create a text generation pipeline using the specified model and tokenizer.
# The 'pipeline' function sets up a ready-to-use text generation pipeline, combining the model and tokenizer.
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text using the pipeline with the specified parameters.
# 'max_new_tokens=256' sets the maximum number of new tokens to generate.
# 'do_sample=True' enables sampling for text generation.
# 'num_beams=1' specifies the number of beams for beam search (1 means no beam search).
# 'temperature=0.3' controls the randomness of predictions by scaling the logits before applying softmax.
# 'top_k=50' considers only the top 50 token predictions for sampling.
# 'top_p=0.95' enables nucleus sampling, considering tokens that have a cumulative probability of 0.95.
# 'max_time=180' sets the maximum generation time to 180 seconds.
outputs = pipe(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    num_beams=1,
    temperature=0.3,
    top_k=50,
    top_p=0.95,
    max_time=180
)

# Print the generated text by stripping out the prompt portion and displaying only the new generated content.
print(outputs[0]['generated_text'][len(prompt):].strip())
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

- **Data set used was:** https://huggingface.co/datasets/gretelai/synthetic_text_to_sql

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

Ignore columns other than "sql_prompt", "sql_context", "sql" from the dataset.

#### Training Hyperparameters

- **Training regime:** Mixed precision (fp16) for efficiency. <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

### Training aftermath

The model was trained on the RTX 3060 OC 12 GB variant. It took 5 hours to train the model with 10,000 values for training and 3,300 values for testing with 2 Epochs.
