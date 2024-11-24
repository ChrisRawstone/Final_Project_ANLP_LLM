# cuda torch test

import torch

# check if cuda is available
print(torch.cuda.is_available())

# get the number of GPUs
print(torch.cuda.device_count())

# get the current device index
print(torch.cuda.current_device())

# get the name of the current device
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# small test
device = torch.device('cuda')
x = torch.rand(5, 3).to(device)
print(x)
