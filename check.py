import torch
print(f"PyTorch Version: {torch.__version__}")
print(torch.cuda.is_available())

print("CUDA version:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")