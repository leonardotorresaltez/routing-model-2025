import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

print(device)
