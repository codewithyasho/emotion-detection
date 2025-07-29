# !pip install transformers datasets torch scikit-learn

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Load your dataset
df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "label"])
df = df.drop_duplicates()

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the text


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Train-test split
split = tokenized_dataset.train_test_split(test_size=0.2)
train_data = split["train"]
test_data = split["test"]

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(le.classes_))

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    logging_steps=10,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate
preds = trainer.predict(test_data)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print(classification_report(y_true, y_pred, target_names=le.classes_))
print("Accuracy score:", accuracy_score(y_true, y_pred))
