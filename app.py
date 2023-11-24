from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import streamlit as st
import torch 

# Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

dataset_name = "c4"
task_name = "en"
dataset = load_dataset(dataset_name, task_name, split="validation",streaming=True)

# count = 0
# for example in dataset:
#     if count >= 1000:
#         break
#     # Process or use the example as needed
#     print(example["text"])  # Or perform any other operation
#     count += 1

#convert dataset to a list so I can fine tune
dataset_list = list(dataset)

# sampleing subset for demonstration purposes
subset_size = 1000
data_subset = dataset_list[:min(subset_size, len(dataset_list))]

# extracting texts from the subset
texts = [example["text"] for example in data_subset]

# tokenize the texts
tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Define custom dataset class
class MLMCustomDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts['input_ids'])
    
    def __getitem__(self, index):
        input_ids = self.tokenized_texts['input_ids'][index].clone()
        labels = input_ids.clone()

        # Randomly mask 15% of tokens for MLM training
        probability_matrix = torch.rand(input_ids.shape)
        mask_indices = probability_matrix < 0.15
        mask_indices[torch.logical_or(input_ids == 101, input_ids == 102)] = False  # Do not mask [CLS] and [SEP] tokens
        input_ids[mask_indices] = 103  # Mask token ID
        
        return {'input_ids': input_ids,
                'attention_mask': self.tokenized_texts['attention_mask'][index],
                'labels': labels}
    
mlm_dataset = MLMCustomDataset(tokenized_texts)

# split the data into training and validation sets
train_indices, val_indices = train_test_split(range(len(mlm_dataset)), test_size=0.2, random_state=42)

# splitting the data into training and validation sets
train_dataset = torch.utils.data.Subset(mlm_dataset, train_indices)
val_dataset = torch.utils.data.Subset(mlm_dataset,val_indices)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./output/fine_tuned_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

results = trainer.evaluate()

st.title("Machine Learning MLM Model Presentation")
st.subheader("Evaluation Metrics")
st.text(results)