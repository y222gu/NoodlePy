import csv
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup,RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return pd.DataFrame(data, columns=['precondition', 'statement', 'label'])
def read_csv2(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return pd.DataFrame(data, columns=['precondition', 'statement'])

# Load the Dataset
train_df = read_csv('data/pnli_train.csv')
dev_df = read_csv('data/pnli_dev.csv')
test_df = read_csv2('data/pnli_test_unlabeled.csv')

# main.ipynb

import csv
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


# Convert labels to integers
train_df['label'] = train_df['label'].astype(int)
dev_df['label'] = dev_df['label'].astype(int)

# Tokenize the Data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def tokenize_function(df):
    return tokenizer(df['precondition'].tolist(), df['statement'].tolist(), padding="max_length", truncation=True)


train_encodings = tokenize_function(train_df)
dev_encodings = tokenize_function(dev_df)
test_encodings = tokenize_function(test_df)


# Create a Dataset Class
class PNliDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


train_dataset = PNliDataset(train_encodings, train_df['label'].tolist())
dev_dataset = PNliDataset(dev_encodings, dev_df['label'].tolist())
test_dataset = PNliDataset(test_encodings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64)

# Define the Model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Move model to CPU
device = torch.device('cuda')
model.to(device)
num_epochs = 10
# Set up the learning rate scheduler
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training and Evaluation Loop
# num_epochs = 3
eval_interval = 400
best_accuracy = 0


def evaluate(model, dev_loader):
    model.eval()
    total, correct = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dev_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

    return accuracy, precision, recall, f1


global_step = 0
model.train()

for epoch in range(num_epochs):
    for step, batch in enumerate(tqdm(train_loader)):
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1

        if global_step % eval_interval == 0:
            accuracy, precision, recall, f1 = evaluate(model, dev_loader)
            print(
                f"Step {global_step}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

            # Save the model if it has the best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_model.pt')

# Load the best model for final evaluation
model.load_state_dict(torch.load('best_model.pt'))

# Final Evaluation on Dev Data
final_accuracy, final_precision, final_recall, final_f1 = evaluate(model, dev_loader)
print(
    f"Final Evaluation: Accuracy = {final_accuracy:.4f}, Precision = {final_precision:.4f}, Recall = {final_recall:.4f}, F1 = {final_f1:.4f}")

# Predict on Test Data
model.eval()
test_loader = DataLoader(test_dataset, batch_size=64)
predictions = []

with torch.no_grad():
    for batch in test_loader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

# Save the Predictions
with open('upload_predictions.txt', 'w') as f:
    for label in predictions:
        f.write(f"{label}\n")

# Load the Dataset
# train_df = pd.read_csv('data/pnli_train.csv', sep='\t', header=None, names=['precondition', 'statement', 'label'])
# dev_df = pd.read_csv('data/pnli_dev.csv', sep='\t', header=None, names=['precondition', 'statement', 'label'])
# test_df = pd.read_csv('data/pnli_test_unlabeled.csv', sep='\t', header=None, names=['precondition', 'statement'])

# Tokenize the Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(df):
    return tokenizer(df['precondition'].tolist(), df['statement'].tolist(), padding="max_length", truncation=True)
# train_df['precondition'].tolist()
# train_df['statement'].tolist()
train_encodings = tokenize_function(train_df)
dev_encodings = tokenize_function(dev_df)
test_encodings = tokenize_function(test_df)

# Create a Dataset Class
class PNliDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = PNliDataset(train_encodings, train_df['label'].tolist())
dev_dataset = PNliDataset(dev_encodings, dev_df['label'].tolist())
test_dataset = PNliDataset(test_encodings)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128)

# main.ipynb
# Define the Model
model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
num_epochs = 3
# Set up the learning rate scheduler
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training and Evaluation Loop

eval_interval = 500
best_accuracy = 0


def evaluate(model, dev_loader):
    model.eval()
    total, correct = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dev_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

    return accuracy, precision, recall, f1


global_step = 0
model.train()

for epoch in range(num_epochs):
    for step, batch in enumerate(tqdm(train_loader)):
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1

        if global_step % eval_interval == 0:
            accuracy, precision, recall, f1 = evaluate(model, dev_loader)
            print(
                f"Step {global_step}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

            # Save the model if it has the best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_model.pt')

# Load the best model for final evaluation
model.load_state_dict(torch.load('best_model.pt'))

# Final Evaluation on Dev Data
final_accuracy, final_precision, final_recall, final_f1 = evaluate(model, dev_loader)
print(
    f"Final Evaluation: Accuracy = {final_accuracy:.4f}, Precision = {final_precision:.4f}, Recall = {final_recall:.4f}, F1 = {final_f1:.4f}")

# Predict on Test Data
model.eval()
test_loader = DataLoader(test_dataset, batch_size=64)
predictions = []

with torch.no_grad():
    for batch in test_loader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

# Save the Predictions
with open('upload_predictions.txt', 'w') as f:
    for label in predictions:
        f.write(f"{label}\n")
