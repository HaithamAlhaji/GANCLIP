import torch

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Print CUDA version
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available.")