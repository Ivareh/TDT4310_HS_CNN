import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel

class BertCNN(nn.Module):
    def __init__(self, num_classes):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        # Pass input_ids and attention_mask to BERT model
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the output of the last layer of the BERT model
        bert_output = bert_output.last_hidden_state.unsqueeze(1)
        # Pass the output through convolutional and pooling layers
        x = self.conv1(bert_output)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        # Flatten the output of the convolutional layers
        x = x.view(-1, 32 * 7 * 7)
        # Pass the output through fully connected layers
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
