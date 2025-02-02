# Spaced Repetition Sampler

Spaced Repetition Sampler is a PyTorch sampler that implements a spaced repetition–based sampling strategy. It prioritizes "hard" samples (i.e. those with high loss) for more frequent review while dynamically increasing the cooldown (interval) for samples that are repeatedly answered correctly. This approach can help improve training efficiency by focusing the model's attention where it is most needed.

The sampler is designed to integrate easily with PyTorch DataLoaders and can also be used with the Hugging Face Trainer for advanced training pipelines.

## Features

- **Dynamic Scheduling:**  
  Samples are prioritized by difficulty, determined via an exponential moving average (EMA) of the training loss.
- **Adaptive Cooldown:**  
  Each sample tracks its own review interval, which grows when a sample is learned easily and resets when it is "missed" (i.e. has higher loss than expected).
- **Flexible Configuration:**  
  Easily adjust parameters such as the base spacing, growth rate, maximum interval, and minimum unseen (backlog) ratio.
- **Easy Integration:**  
  Works out of the box with PyTorch DataLoaders and can be incorporated into Hugging Face Trainer workflows.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/spaced-repetition-sampler.git
cd spaced-repetition-sampler
```

2. (Optional) Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows use: venv\Scripts\activate
```

3. Install Dependencies

At a minimum, you will need PyTorch:

```bash
pip install torch
```

If you plan to use the Hugging Face integration, install the Transformers library as well:

```bash
pip install transformers
```

(Additional dependencies may be added as needed.)

## Example Usage

Below is a minimal example demonstrating how to use the Spaced Repetition Sampler with a simple PyTorch training loop.

### Example: Training Loop with PyTorch DataLoader

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from spaced_repetition_sampler import SpacedRepetitionSampler, ExponentialMovingStats

# --- Define a toy dataset ---
class ToyDataset(Dataset):
    def __init__(self, num_samples=200):
        self.data = torch.randn(num_samples, 10)
        self.labels = torch.randint(0, 5, (num_samples,))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Wrap the dataset so that each sample includes its index.
class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x, y, idx

# A collate function that organizes the batch.
def collate(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    idxs = [item[2] for item in batch]
    return torch.stack(xs, 0), torch.tensor(ys), idxs

# --- Create the dataset and sampler ---
base_dataset = ToyDataset(num_samples=200)
indexed_dataset = IndexedDataset(base_dataset)
loss_stats = ExponentialMovingStats(alpha=0.01)

# Instantiate the Spaced Repetition Sampler.
sampler = SpacedRepetitionSampler(
    data_source=indexed_dataset,
    loss_stats=loss_stats,
    num_steps=50,
    batch_size=16,
    spacing_steps=5,           # Base spacing in training steps.
    min_backlog_ratio=0.1,     # At least 10% of each batch from unseen samples.
    growth_rate=1.2,           # Increase interval by 20% on success.
    max_interval=100,
    success_threshold_std=0.5  # A sample is considered "easy" if loss < mean + 0.5*std.
)

# Create a DataLoader with the custom sampler.
data_loader = DataLoader(
    indexed_dataset,
    batch_sampler=sampler,
    collate_fn=collate_with_indices,
)

# --- Define a simple model, loss, and optimizer ---
model = nn.Sequential(nn.Linear(10, 5))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# --- Training Loop ---
for step, (inputs, labels, indices) in enumerate(data_loader):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Update the sampler with the batch loss.
    sampler.update_difficulties(loss.item())

    print(f"Step {step}, Loss: {loss.item()}")
```

### Example: Integration with Hugging Face Trainer

If you prefer using the Hugging Face Trainer, you can integrate the sampler as follows:

```python
from transformers import Trainer, TrainingArguments, CollateForWhateverTask
from datasets import load_dataset
from torch.utils.data import IndexedDataset
from spaced_repetition_sampler import SpacedRepetitionSampler, ExponentialMovingStats

# Prepare your dataset and model as before.
dataset = load_dataset(...)
loss_stats = ExponentialMovingStats(alpha=0.01)

sampler = SpacedRepetitionSampler(
    data_source=dataset,
    loss_stats=loss_stats,
    num_steps=50,
    batch_size=16,
    spacing_steps=5,
    min_backlog_ratio=0.1,
    growth_rate=1.2,
    max_interval=100,
    success_threshold_std=0.5,
)

# Define training arguments.
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    max_steps=50,
    remove_unused_columns=False,  # Ensure 'indices' field is retained.
)

# Use the custom Trainer (e.g., SRSTrainer) provided in this project.
trainer = SRSTrainer(
    model=model,
    args=training_args,
    train_dataset=indexed_dataset,
    data_collator=CollateForWhateverTask(),
    train_sampler=sampler,
)
trainer.train()
```

(Ensure that you have the custom Trainer implementation available in your project if using the Hugging Face integration.)

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Dustin Ewan
