import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
print(input.size())
loss = nn.CrossEntropyLoss()
target = torch.empty(3, dtype=torch.long).random_(5)
print(input)
print(target.size())
print(loss(input,target))