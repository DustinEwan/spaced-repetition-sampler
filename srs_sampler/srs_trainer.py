import torch
import torch.nn as nn
from transformers import Trainer
from torch.utils.data import DataLoader
from typing import Dict, Union, Any

class SpacedRepititionTrainer(Trainer):
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        loss_stats = ExponentialMovingStats()

        num_steps = self.args.max_steps
        # Take first element if num_steps is a tuple
        if isinstance(num_steps, tuple):
            num_steps = num_steps[0]
        

        train_sampler = SpacedRepetitionSampler(
            self.train_dataset, loss_stats, num_steps, self.args.train_batch_size
        )

        self.train_sampler = train_sampler

        return DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        loss = super().training_step(model, inputs)

        nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm, norm_type=2.0)

        # Update the sampler with the loss from this batch
        self.train_sampler.update_difficulties(loss.item())

        return loss
