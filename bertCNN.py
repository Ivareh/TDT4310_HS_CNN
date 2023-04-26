import gc
import torch
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from config import readconfig


print('attention_probs_dropout_prob: ', readconfig.get_bert_config_value('attention_probs_dropout_prob'))
  
class BertCNN(nn.Module):
    def __init__(self, num_classes, dropout_prob):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(477984, 32)
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        # all_layers  = [13, 32, 64, 768]
        x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in hidden_states if isinstance(t, torch.Tensor)]), 0), 0, 1)
        x = self.pool(self.dropout1(self.relu(self.conv1(self.dropout1(x)))))
        x = self.fc(self.dropout1(self.flat(self.dropout1(x))))
        return self.softmax(x)
