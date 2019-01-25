import torch
import torch.nn as nn
import numpy as np

num = torch.tensor([1.2, 2, 3,5])
mask = torch.tensor([0,1,1,0])
num.masked_fill(mask, torch.tensor(3.0))
print(num)