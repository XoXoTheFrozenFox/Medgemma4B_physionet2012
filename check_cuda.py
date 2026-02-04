import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print("basic cuda matmul OK; y.mean:", y.mean().item())
