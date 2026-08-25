import torch


class AutoCast(torch.nn.Module):
    def __init__(self, module, dtype=torch.bfloat16):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=self.dtype):
            return self.module(*args, **kwargs)
