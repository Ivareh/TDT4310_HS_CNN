import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from config import readconfig


print('attention_probs_dropout_prob: ', readconfig.get_bert_config_value('attention_probs_dropout_prob'))
  
class BertCNN(nn.Module):
    def __init__(self, num_classes, dropout_prob):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=13, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(442, 3)
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)

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
        print('x.shape: ', x.shape)
        x = x.view(x.size(0), -1)
        # Pass the output through fully connected layers
        x = self.dropout1(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
