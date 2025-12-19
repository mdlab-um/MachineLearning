import os
import kagglehub
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
import re
import random
from pytorch_utils import *


##### Download and Read in the raw dataset
# Download the dataset (returns the folder path)
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
csv_path = os.path.join(path, "IMDB Dataset.csv")
df = pd.read_csv(csv_path)

print("Dataset downloading: success!")
print(f"Path to dataset: {os.path.join(path, "IMDB Dataset.csv")}")
print(df)
print(df.info())


###### Tokenization
# 1. Read in dataset
dataset = Dataset.from_csv(csv_path)
# 2. Initialize the Tokenizer
# DistilBERT is a great balance of speed and accuracy
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 3. Define the Cleaning & Tokenizing Function
def preprocess_function(examples):
    # 'examples' is a dictionary containing a BATCH of 1000 reviews

    # A. Clean HTML tags for the whole batch at once
    # We use a list comprehension here which is very fast
    cleaned_texts = [re.sub(r'<.*?>', ' ', text) for text in examples['review']]

    # B. Tokenize the batch
    # truncation=True: Cut reviews longer than 512 tokens (model limit)
    # padding="max_length": Pad shorter reviews with 0s so they are all equal length
    return tokenizer(cleaned_texts, truncation=True, padding="max_length", max_length=512)

# 4. Apply to the whole dataset efficiently
# batched=True is the secret sauce here. It processes 1000 rows at a time.
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# --- Inspect the Result ---
print(tokenized_dataset)
# You will now see new columns: 'input_ids', 'attention_mask'

# A quick sanity check to see if the tokenization has done a proper job
print("Sanity check using one random example:")
idx = random.randint(0, len(dataset)-1)
samplex = tokenized_dataset[idx]['input_ids']
print(f"Draw a random sample: index={idx}")
print(samplex)
print("pre-processed text: ")
print(tokenizer.decode(samplex))


###### Preprocessing labels
# Create a simple mapping function
def map_labels(example):
    return {"label": 1 if example["sentiment"] == "positive" else 0}

tokenized_dataset = tokenized_dataset.map(map_labels)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# 1. First Split: Train (70%) vs The Rest (30%)
# We use test_size=0.3 to set aside 30% for Val+Test
split_1 = tokenized_dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = split_1['train']
temp_dataset = split_1['test'] # This contains the remaining 30%

# 2. Second Split: Validation (15%) vs Test (15%)
# We split the temp_dataset in half (0.5 of 30% = 15%)
split_2 = temp_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = split_2['train']
test_dataset = split_2['test']

# --- Verify the sizes ---
total_size = len(tokenized_dataset)
print(f"Total: {total_size}")
print(f"Train: {len(train_dataset)} ({len(train_dataset)/total_size:.0%})")
print(f"Val:   {len(val_dataset)} ({len(val_dataset)/total_size:.1%})")
print(f"Test:  {len(test_dataset)} ({len(test_dataset)/total_size:.1%})")


###### Dataloader
# from torch.utils.data import DataLoader
#
# seed = 42
# BATCH_SIZE = 16
#
# set_seed(seed)
#
# # 1. Train Loader (Shuffle = True)
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )
#
# # 2. Validation Loader (Shuffle = False)
# # Used for checking loss during training to prevent overfitting
# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )
#
# # 3. Test Loader (Shuffle = False)
# # Used only ONCE at the very end to report final accuracy
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )
