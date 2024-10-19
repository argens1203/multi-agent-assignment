import torch

side = 5
action_size = 5
state_size = 5 * 5 * 3 + 1
device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
dtype = torch.float32
# if device.type == "mps" else torch.float64
debug = False
