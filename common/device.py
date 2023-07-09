import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.has_mps:
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
