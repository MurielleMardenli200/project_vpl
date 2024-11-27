#!/usr/bin/env python3
import os
import random
from constants.task_type import TASK_CONFIG_1
from constants.wandb_constants import WANDB_RUN_NAME
from generate_base import ResNetWithRNN, collate_fn, set_seed
import torch
import torch.nn as nn
import numpy as np

import matplotlib

# Force to not use a GUI for plots
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb
from database.coco_1 import generate_test_transforms, COCO_RNN
from torchvision.models.resnet import resnet50

# Wandb login

os.environ["WANDB_API_KEY"] = "88ac16dc79bae11217c166cadfc8397f9bb473f9"
os.environ["WANDB_ENTITY"] = "murielle-mardenli"


def train(config=None):
    with wandb.init(name=WANDB_RUN_NAME, config=config):
        config = wandb.config

        configs_tasks = TASK_CONFIG_1
        train_task_config = configs_tasks[0]
        IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN = generate_test_transforms(
            train_task_config
        )
        train_dataset = COCO_RNN(
            "/Users/muriellemardenli/Desktop/mainn/ResearchProjects/SNAIL/shared1000/",
            transforms=IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN,
            same_pair_probability=0.5,
            same_not_rand=True,
            idx_ref=4,
            idx_other=5,
            num_scrambled=3,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        model = ResNetWithRNN(
            hidden_size=config.hidden_size, output_size=1, num_layers=config.num_layers
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )

        all_seqs = []
        total_iterations = 0

        for epoch in range(config.num_epochs):
            epoch_losses = []
            # print(f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (sequences, labels, same_pairs) in enumerate(train_loader):
                total_iterations += 1
                same_pair = same_pairs[1]
                num_seq = sequences.shape[1]
                batch_size = sequences.shape[0]

                optimizer.zero_grad()
                outputs, seqs = model(sequences.permute(1, 0, 2, 3, 4))
                print("outputs")
                print(outputs)

                outputs_loss = outputs.reshape(num_seq, batch_size, -1)[-1]

                loss = criterion(outputs_loss.squeeze(), same_pairs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
                all_seqs.append(outputs.detach().cpu())

                torch_seq = torch.stack(all_seqs, dim=0)
                iteration = len(all_seqs) - 1
                seq = torch_seq[iteration].reshape(num_seq, batch_size, -1)

                wandb.log({"batch_loss": loss.item()})

                plt.figure(figsize=(10, 6))
                for i in range(sequences.shape[0]):
                    plt.plot(
                        seq[:, i].detach().cpu().numpy(),
                        label=f"batch index {i}",
                        marker="o",
                    )
                plt.legend()

                plt.title(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Iteration {total_iterations}"
                )
                plt.ylabel("RNN output")
                plt.xlabel("Sequences")
                plt.savefig("temp_plot.png")
                plt.grid(True)
                plt.close()
                print(f"output for iteration {total_iterations}")

                if batch_idx % 5 == 0:
                    wandb.log({"RNN_output_plot": wandb.Image("temp_plot.png")})

                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({"learning_rate": current_lr})

            epoch_loss = np.mean(epoch_losses)
            wandb.log({"epoch_loss": epoch_loss})
        torch.save(model.state_dict(), "model.pth")
        wandb.save("model.pth")


sweep_config = {
    "optimizer": "SGD",
    "method": "grid",
    "metric": {"name": "epoch_loss", "goal": "minimize"},
    "parameters": {
        "hidden_size": {"values": [256, 512, 1024]},
        "batch_size": {"values": [5, 10, 20]},
        "learning_rate": {"values": [0.001, 0.005, 0.01]},
        "num_layers": {"values": [3, 4, 5]},
        # "optimizer": {"values": ["SGD", "Adam"]},
        "weight_decay": {"values": [1e-5]},
        # TODO: Increase on GPU
        "num_epochs": {"values": [3]},
    },
}

if __name__ == "__main__":
    set_seed()
    sweep_id = wandb.sweep(sweep_config, project="RNN-HyperparamSearch")
    wandb.agent(sweep_id, train)
