# Finetune your own model


This notebook aims to walk you through all the steps needed to finetune your own transformer using the Huggingface `transformers` ecosystem. 

### Motivation
Why finetune a transformer? Finetuning is useful whenever you wish to have a specialized model for a specific task. Training competitively-sized transformers from scratch these days requires an immense amount of resources, both in data and in compute. On the other hand, prompt engineering is very fast, but has a limited scope and can't really teach the model entirely new concepts. Finetuning attempts to strike a middle ground for tasks that still, for example, require an understanding of language, but require more specialization than prompt engineering can provide.

### Setup and Considerations
Before finetuning a transformer, you have to decide what to finetune it *on*. What you choose depends on what kind of task you want it to accomplish. In this case, I will be finetuning DistilGPT-2 on Shakespeare's sonnets, but you can pick anything you would like. It is also possible to finetune, say, image generation models on a set of images, but the code here would need to be modified slightly. 

One consideration that is a fundamental issue in finetuning is the tradeoff between catastrophic forgetting and specialization. That is, by finetuning it on new data, the model may forget parts of what it learned when it was originally trained. Different strategies can be used to mitigate this and find the right tradeoff. 

---

With that out of the way, let's get to some code!

Import the necessary libraries:


```python
#@title Import and install necessary libraries
import sys, os
if sys.prefix != sys.base_prefix or "google.colab" in sys.modules or os.environ.get("BINDER_SERVICE_HOST") is not None:
    pass
else:
    print("Not in a virtual environment. Please create one and activate it before running this script.")
    exit()


os.system("pip install 'accelerate>=0.26.0' 'transformers[torch]' datasets torch")
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
```

### Data loading

The first step to training any model is to prepare the data. Often, when working with custom datasets, this is the step that takes the most work! In this case, we are using a pre-built dataset, so Huggingface lets us load this in automatically. Here, I've provided an example of how you might further process individual data samples and use `.map` to apply it. A slightly different, more declarative strategy than the Pytorch paradigm of creating a custom Dataset class with lots of boilerplate.


```python

def load_and_preprocess_data():
    # Load the dataset from Hugging Face
    sonnets_dataset = load_dataset("kkawamu1/shakespeares_sonnets")
    
    # Create train/validation split (90/10)
    # If you have an extremely limited dataset, you could employ other methods such as k-fold cross validation. 
    shuffled_dataset = sonnets_dataset['train'].shuffle(seed=42)
    train_size = int(0.9 * len(shuffled_dataset))
    
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))
    
    print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Example of preprocessing: lowercasing and stripping whitespace
    def clean_text(example):
        # Crude insertion of <|eos|>. In reality, the token should probably be explicitly added to the tokenizer. You don't have to worry about this for now.
        example["text"] = example["text"].lower().strip() + "<|eos|>"

        return example
    
    train_dataset = train_dataset.map(clean_text)
    val_dataset = val_dataset.map(clean_text)
    
    return train_dataset, val_dataset

```

### Further Data Preparation

Transformer models, internally, receive and generate tokens. Thus, we need to tokenize our dataset. Here, a premade tokenizer comes with `distilgpt2`, so we can use that. 


```python

def prepare_for_training(train_dataset, val_dataset):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Has to be done for distilgpt2 in order to ensure there exists a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the data (just calls tokenizer.encode on each text)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["id"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["id"])
    
    # Set the format for PyTorch
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask"])
    
    return tokenizer, tokenized_train, tokenized_val
```

### Model Training

Now for the most exciting part, training the model! Fortunately, HF does a lot of the work for us, so this is mostly about specifying hyperparameters. 


```python
def train_model(tokenizer, train_dataset, val_dataset):

    # Loads the pre-trained DistilGPT-2 model
    # Other great pretrained models exist, depending on how much compute you have
    # A list to get you started: https://github.com/huggingface/transformers/blob/70e57e4710d8a617a6f0ea73183d9bc4c91063c9/src/transformers/models/auto/modeling_auto.py#L559
    # But any model on the Hub works
    model = AutoModelForCausalLM.from_pretrained("distilgpt2") 
    

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5, # Train for 5 epochs
        per_device_train_batch_size=4, # Batch size of 4. If you have more GPU memory, you can increase this
        per_device_eval_batch_size=4,
        eval_steps=100,
        save_steps=100,
        warmup_steps=100,
        eval_strategy="steps",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-5, # Typical starting point for fine-tuning
        save_total_limit=2,  # Keep only the last 2 models
        weight_decay=0.01,
        fp16=bool(torch.cuda.is_available()),
    )
    
    # Data collator. Think of this like a pytorch DataLoader, but from Hugging Face.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Initialize the Trainer. Pass in model, arguments, collator, and datasets.
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    print("Starting training...")
    # Huggingface does the rest!
    trainer.train()
    # If resuming from a checkpoint
    # trainer.train(resume_from_checkpoint=True)
    
    return model, trainer

# Utility function demonstrating `save_pretrained`
MODEL_PATH = "./shakespeare_sonnets_model"
def save_model(model, tokenizer):
    output_dir = MODEL_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir
```

### Generation

After training, we can now use our model to generate samples. Here, you need to tokenize the input, run it through the model and, decode the output back into text.


```python

def generate(prompt="Shall I compare thee ", max_length=250):
    try:
        # Load model and tokenizer from specified path
        model_dir = MODEL_PATH
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Set pad token, again
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        inputs = tokenizer(prompt, return_tensors="pt") #Tokenize input prompt
        
        outputs = model.generate( # Generate output. Huggingface handles the internals.
            inputs.input_ids,
            max_length=max_length,
            temperature=0.9,
            top_p=0.92,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # Decode the generated tokens

        generated_text = generated_text[len(prompt):].split("<|eos|>")[0].strip()
        
        return generated_text
    
    except Exception as e:
        print(f"Error generating sonnet: {e}")
        return "Could not generate sonnet. Make sure the model has been trained and saved."
```

### Running

Run the following cell to run the code and see the output!


```python
def main():

    train_dataset, val_dataset = load_and_preprocess_data()

    tokenizer, tokenized_train, tokenized_val = prepare_for_training(train_dataset, val_dataset)

    model, trainer = train_model(tokenizer, tokenized_train, tokenized_val)

    model_dir = save_model(model, tokenizer)

    print("\nGenerating a sample sonnet...")
    sample_sonnet = generate("Forsooth for I shall not")
    print(sample_sonnet)
    # Output may vary
    """
        depart
        with a sweet night,
        for the beauty of my mind will see;
        for this summer is no year in the summer
        which lies on the end.
        but i will stay with those that have my ear,
        and my heart to my heart where love is,
        when that night shall show
        as if i do not renew my life:
        as with a summer which brings for myself,
        it knows not where i dwell, and keeps me alive.
    """

if __name__ == "__main__":
    main()

```

### Closing

You've now seen how to put the pieces together. It really can be more simple than you think! Now go forth and finetune your own wild creations.

-JBlitzar

Original source and licensing information at [https://github.com/JBlitzar/finetune-your-own-model](https://github.com/JBlitzar/finetune-your-own-model)
