import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import AdamW, AutoModel
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from bertCNN import BertCNN
from preprocessingdata import load_process_data

# Define the training parameters
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
    train_data, sampler=train_sampler, batch_size=batch_size)

# loss function
cross_entropy = torch.nn.NLLLoss()

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

valid_loader = DataLoader(
    val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_seq, test_mask, test_y)

test_sampler = SequentialSampler(test_data)

test_loader = DataLoader(
    test_seq, sampler=test_sampler, batch_size=batch_size)

# train the model
def train():
    # set the model to train mode
    model.train()
    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over the training dataloader
    for i, batch in enumerate(train_loader):
        step = i+1
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        del batch
        gc.collect()
        torch.cuda.empty_cache()
        # clear previously calculated gradients
        model.zero_grad()
        # forward pass
        print("FORWARD SHAPES:")
        print(sent_id)
        preds = model(sent_id.to(device).long(), mask)
        
        # compute the loss
        loss = loss_fn(preds, labels)

        total_loss += float(loss.item())

        # backward pass and update the weights
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        # append the model predictions
        total_preds.append(preds.detach().cpu().numpy())

        gc.collect()
        torch.cuda.empty_cache()

        

        # compute the training loss of the epoch
        avg_loss = total_loss / (len(train_loader)*batch_size)

        # predictions are in the form of (no. of batches, size of batch, no. of classes).
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        # returns the loss and predictions
    return avg_loss, total_preds

def evaluate():
    print("\n\nEvaluating...")
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    # empty list to save the model predictions
    total_preds = []
    # iterate over batches
    total = len(valid_loader)
    for i, batch in enumerate(valid_loader):
        step = i+1
        percent = "{0:.2f}".format(100 * (step / float(total)))
        lossp = "{0:.2f}".format(total_loss/(total*batch_size))
        filledLength = int(100 * step // total)
        bar = '█' * filledLength + '>' * (filledLength < 100) + '.' * (99 - filledLength)
        print(f'\rBatch {step}/{total} |{bar}| {percent}% complete, loss={lossp}, accuracy={total_accuracy}', end='')
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        del batch
        gc.collect()
        torch.cuda.empty_cache()
        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)
            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)
            total_loss += float(loss.item())
            total_preds.append(preds.detach().cpu().numpy())

    gc.collect()
    torch.cuda.empty_cache()
    # compute the validation loss of the epoch
    avg_loss = total_loss / (len(valid_loader)*batch_size)
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# set initial loss to infinite
best_valid_loss = float('inf')

epochs = 3
current = 1
# for each epoch
while current <= epochs:

    print(f'\nEpoch {current} / {epochs}:')

    # train model
    train_loss, _ = train()

    # evaluate model
    valid_loss, _ = evaluate()

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

    print(f'\n\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

    current = current + 1

# get predictions for test data
gc.collect()
torch.cuda.empty_cache()

with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()


print("Performance:")
# model's performance
preds = np.argmax(preds, axis=1)
print('Classification Report')
print(classification_report(test_y, preds))

print("Accuracy: " + str(accuracy_score(test_y, preds)))