import torch

side = 4
state_size = 4 * 4 * 3
device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
dtype = torch.float32 if device.type == "mps" else torch.float64
