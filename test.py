import torch
from torchmetrics import F1Score

f1 = F1Score(task='multiclass',num_classes=3,average='none')

x = torch.tensor([0,1,2,1,1])
y = torch.tensor([0,1,1,2,1])

print(f'f1{f1(x,y)}')