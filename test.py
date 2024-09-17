import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
    print("Number of GPU devices:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("GPU is not available")
