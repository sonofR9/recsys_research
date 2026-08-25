# NeuralRec

A small, modular deep learning training framework for PyTorch with built-in support for distributed training, mixed precision, and extensible callbacks.

## Features

- **Training Runner** — Simple training loop with callback hooks for customization
- **Distributed Training** — First-class DDP support with helper utilities
- **Mixed Precision** — AutoCast wrapper for automatic bf16/fp16 training
- **Callbacks** — Modular system for logging, validation, gradient clipping, and TensorBoard
- **Data Loading** — Extended DataLoader with transform pipelines and GPU prefetching
- **Neural Networks** — Transformer encoders with optional FlashAttention support

## Installation

```bash
# Basic installation
pip install -e .

# With development tools (ruff, mypy, pytest)
pip install -e ".[dev]"

# With FlashAttention support
pip install -e ".[flash-attn]"

# With TensorBoard logging
pip install -e ".[tensorboard]"

# With HuggingFace datasets/transformers
pip install -e ".[huggingface]"

# Install everything
pip install -e ".[all]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.10.0
- NumPy >= 2.2.0

## Quick Start

```python
from neuralrec.run.train import TrainRunner
from neuralrec.run.callbacks import LoggingCallback, GradientNormClippingCallback
from neuralrec.data.dataloader import DataLoader

# Define your model (must return dict with 'loss' key)
model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
train_loader = DataLoader(dataset, batch_size=32)

TrainRunner(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    callbacks=[
        GradientNormClippingCallback(max_norm=1.0),
        LoggingCallback().every_n_steps(10),
    ],
).fit(num_epochs=10)
```

## Modules

### Training (`neuralrec.run`)

**TrainRunner** — Core training loop with callback support:

```python
from neuralrec.run.train import TrainRunner

runner = TrainRunner(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    callbacks=[...],
)
runner.fit(num_epochs=10)
```

**Distributed utilities** — Initialize and manage distributed training:

```python
from neuralrec.run.distributed import init_process_group, destroy_process_group, is_chief

init_process_group()  # Auto-selects NCCL for GPU, Gloo for CPU
# ... training code ...
destroy_process_group()
```

### Callbacks (`neuralrec.run.callbacks`)

All callbacks support chaining with `.every_n_steps(n)` and `.ignore_if(condition)`:

| Callback | Description |
|----------|-------------|
| `LoggingCallback` | Prints loss and step info |
| `ValidationCallback(loader)` | Runs validation on a separate loader |
| `GradientNormClippingCallback(max_norm)` | Clips gradients before optimizer step |
| `TensorBoardCallback` | Logs metrics to TensorBoard |

```python
from neuralrec.run.callbacks import ValidationCallback, LoggingCallback

callbacks = [
    ValidationCallback(valid_loader).every_n_steps(100).ignore_if(not is_chief()),
    LoggingCallback().every_n_steps(10).ignore_if(not is_chief()),
]
```

### Data (`neuralrec.data`)

**DataLoader** — Extended PyTorch DataLoader with transform support:

```python
from neuralrec.data.dataloader import DataLoader, PrefetchDataLoader
from neuralrec.data.transforms import ToDevice

loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    transforms=ToDevice(torch.device("cuda")),
)

# For async GPU prefetching:
loader = PrefetchDataLoader(
    dataset=dataset,
    batch_size=32,
    device="cuda",
    buffer_size=2,
)
```

**Transforms**:

| Transform | Description |
|-----------|-------------|
| `ToNumpy` | Convert lists to numpy arrays |
| `ToTorch` | Convert lists/numpy to torch tensors |
| `ToDevice(device)` | Move tensors to specified device |

### Neural Networks (`neuralrec.nn`)

**TransformerEncoder** — Standard PyTorch-based transformer:

```python
from neuralrec.nn.transformer import TransformerEncoder

encoder = TransformerEncoder(
    d_model=256,
    nhead=8,
    num_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    causal=True,  # For autoregressive models
)
```

**FlashAttention TransformerEncoder** — Drop-in replacement using FlashAttention:

```python
from neuralrec.nn.flashattn_transformer import TransformerEncoder
# Same interface as above
```

**AutoCast** — Wrapper for automatic mixed precision:

```python
from neuralrec.nn.autocast import AutoCast

model = AutoCast(model, dtype=torch.bfloat16)
```

## Example: Distributed Training

See `examples/train_ddp.py` for a complete example training a recommendation transformer with DDP:

```bash
torchrun --nproc_per_node=4 examples/train_ddp.py
```

```python
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from neuralrec.run.distributed import init_process_group, destroy_process_group, is_chief
from neuralrec.nn.autocast import AutoCast

init_process_group()

train_loader = DataLoader(
    dataset=train_dataset,
    sampler=DistributedSampler(train_dataset),
    ...
)

model = MyModel().cuda()
model = AutoCast(model)
model = DDP(model)

TrainRunner(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    callbacks=[
        GradientNormClippingCallback(1.0),
        LoggingCallback().every_n_steps(10).ignore_if(not is_chief()),
    ],
).fit(num_epochs=10)

destroy_process_group()
```

## Custom Callbacks

Extend the `Callback` base class to create custom callbacks:

```python
from neuralrec.run.callbacks import Callback

class CheckpointCallback(Callback):
    def on_epoch_end(self, runner):
        torch.save(runner.model.state_dict(), f"checkpoint_{runner.current_epoch}.pt")
    
    def on_step_end(self, runner, batch, out):
        # Access loss via out["loss"]
        pass
```

Available hooks:
- `on_train_begin(runner)`
- `on_train_end(runner)`
- `on_epoch_end(runner)`
- `on_step_begin(runner, batch)`
- `on_step_end(runner, batch, out)`
- `on_before_optimizer_step(runner)`