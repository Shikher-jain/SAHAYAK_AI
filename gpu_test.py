import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)
print("GPU:", torch.cuda.get_device_name(0) if device=="cuda" else "None")

# Force computation
x = torch.rand(10000, 10000).to(device)
y = torch.mm(x, x)

print("Done computation")