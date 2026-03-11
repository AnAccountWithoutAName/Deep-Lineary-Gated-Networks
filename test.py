import torch


a = torch.ones(32,6,6)
a = a.repeat(3,1,1,1)
print(a.size())
