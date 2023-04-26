import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import AdamW, AutoModel
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from bertCNN import BertCNN
from preprocessingdata import load_process_data

# Define the training parameters
epochs = 3
num_classes = 7
batch_size = 32
learning_rate = 2e-5
warmup_ratio = 0.1
dropout_prob = 0.1

# From https://github.com/ZeroxTM/BERT-CNN-Fine-Tuning-For-Hate-Speech-Detection-in-Online-Social-Media/blob/7a8c4bc46122465489b58bdf105e961b76bb8a9a/BertCNN.py
input_ids, attention_masks, labels = load_process_data()
df = pd.DataFrame(list(zip(input_ids, attention_masks)), columns=['input_ids', 'attention_masks'])

# Split into training and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(df, labels,
                             random_state=2018, test_size=0.2, stratify=labels)

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                         random_state=2018, test_size=0.5, stratify=temp_labels)


# Load the tokenizer and encode the data
bert = AutoModel.from_pretrained('bert-base-uncased')

# Tokenization
# for train set
train_seq = torch.tensor(train_text['input_ids'].tolist())
train_mask = torch.tensor(train_text['attention_masks'].tolist())
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(val_text['input_ids'].tolist())
val_mask = torch.tensor(val_text['attention_masks'].tolist())
val_y = torch.tensor(val_labels.tolist())

# for test set
test_seq = torch.tensor(test_text['input_ids'].tolist())
test_mask = torch.tensor(test_text['attention_masks'].tolist())
test_y = torch.tensor(test_labels.tolist())

# Define the model
model = BertCNN(num_classes=num_classes, dropout_prob=dropout_prob)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Define the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Define the loss function and accuracy metric
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)
accuracy_fn = Accuracy(task="multiclass", num_classes=num_classes)

# Create the training DataLoader
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

train_loader = DataLoader(
    train_seq, batch_size=batch_size, shuffle=True)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

valid_loader = DataLoader(
    val_seq, batch_size=batch_size)

test_data = TensorDataset(test_seq, test_mask, test_y)

test_sampler = SequentialSampler(test_data)

test_loader = DataLoader(
    test_seq, batch_size=batch_size)

# train the model
num_epochs = 10
for epoch in range(num_epochs):
    # set the model to train mode
    model.train()

    # iterate over the training dataloader
    for i, batch in enumerate(train_loader):
        step = i+1
        print(batch)
        
        # forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # compute the loss
        loss = loss_fn(outputs.logits, labels)

        # backward pass and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # set the model to evaluation mode
    model.eval()

    # track the validation loss and accuracy
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
        # iterate over the validation dataloader
        for batch in valid_loader:
            # move the batch to the appropriate device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # compute the loss and accuracy
            loss = loss_fn(outputs.logits, labels)
            valid_loss += loss.item() * input_ids.size(0)
            valid_acc += (outputs.logits.argmax(1) == labels).sum().item()

    # normalize the validation metrics by the size of the validation set
    valid_loss /= len(val_data)
    valid_acc /= len(val_data)

    # print the validation metrics
    print(f"Epoch [{epoch+1}/{num_epochs}], Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

# evaluate the model on the test set
test_loss = 0.0
test_acc = 0.0
with torch.no_grad():
    # iterate over the test dataloader
    for batch in test_loader:
        # move the batch to the appropriate device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # compute the loss and accuracy
        loss = loss_fn(outputs.logits, labels)
        test_loss += loss.item() * input_ids.size(0)
        test_acc += (outputs.logits.argmax(1) == labels).sum().item()


# normalize the test metrics by the size of the test set
test_loss /= len(test_dataset)
test_acc /= len(test_dataset)

# print the test metrics
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")