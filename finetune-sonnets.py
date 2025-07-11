#!/usr/bin/env python3
"""
Fine-tune DistilGPT-2 on Shakespeare's sonnets for sonnet generation.
"""

import os
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

def load_and_preprocess_data():
    # Load the dataset from Hugging Face
    sonnets_dataset = load_dataset("kkawamu1/shakespeares_sonnets")
    
    # Create train/validation split (90/10)
    shuffled_dataset = sonnets_dataset['train'].shuffle(seed=42)
    train_size = int(0.9 * len(shuffled_dataset))
    
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))
    
    print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Example of preprocessing: lowercasing and stripping whitespace
    def clean_text(example):
        example["text"] = example["text"].lower().strip()
        return example
    
    train_dataset = train_dataset.map(clean_text)
    val_dataset = val_dataset.map(clean_text)
    
    return train_dataset, val_dataset

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

def train_model(tokenizer, train_dataset, val_dataset):

    # Loads the pre-trained DistilGPT-2 model
    # Other great pretrained models exist, depending on how much compute you have
    # A list to get you started: https://github.com/huggingface/transformers/blob/70e57e4710d8a617a6f0ea73183d9bc4c91063c9/src/transformers/models/auto/modeling_auto.py#L559
    # But any model on the Hub works
    model = AutoModelForCausalLM.from_pretrained("distilgpt2") 
    

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=100,
        save_steps=100,
        warmup_steps=100,
        evaluation_strategy="steps",
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
    
    return model, trainer

def save_model(model, tokenizer):
    output_dir = "./shakespeare_sonnets_model"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

def generate_sonnet(prompt="Write a sonnet about ", max_length=250):
    try:
        # Load model and tokenizer
        model_dir = "./shakespeare_sonnets_model"
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
        
        return generated_text
    
    except Exception as e:
        print(f"Error generating sonnet: {e}")
        return "Could not generate sonnet. Make sure the model has been trained and saved."

def main():

    train_dataset, val_dataset = load_and_preprocess_data()

    tokenizer, tokenized_train, tokenized_val = prepare_for_training(train_dataset, val_dataset)

    model, trainer = train_model(tokenizer, tokenized_train, tokenized_val)

    model_dir = save_model(model, tokenizer)

    print("\nGenerating a sample sonnet...")
    sample_sonnet = generate_sonnet("Write a sonnet about love: ")
    print(sample_sonnet)

if __name__ == "__main__":
    main()
