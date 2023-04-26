import torch
from bertCNN import BertCNN
print(torch.zeros(1).cuda())

# Create an instance of the model
model = BertCNN(num_classes=3, dropout_prob=0.1)

# Load an example input
input_ids = torch.tensor([[101, 2023, 2003, 1037, 2024, 2172, 102]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])

# Run the model on the input
output = model(input_ids, attention_mask)

# Print the output
print(output)
