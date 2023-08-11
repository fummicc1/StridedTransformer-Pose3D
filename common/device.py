import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("cuda is used!")
elif torch.has_mps:
    DEVICE = torch.device('mps')
    print("mps is used!")
else:
    DEVICE = torch.device('cpu')
    print("cpu is used...")
