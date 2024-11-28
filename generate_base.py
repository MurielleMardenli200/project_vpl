#!/usr/bin/env python3
import random
import sys
import os
from constants.task_type import TASK_CONFIG_1
from constants.wandb_constants import WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from database.coco_1 import create_transform_aperature, generate_test_transforms
from database.coco_1 import COCO_RNN
from torchvision.models.resnet import resnet50
import argparse

import wandb


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_layers_except_last(net: nn.Module, last_layer_name: str):
    if isinstance(last_layer_name, str):
        last_layer_name = [last_layer_name]
    for name, param in net.named_parameters():
        param.requires_grad = any(
            trainable_name in name for trainable_name in last_layer_name
        )


class ResNetWithRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=2):
        super(ResNetWithRNN, self).__init__()
        resnet = torch.load("resnet50_pretrained.pth")
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.rnn = nn.RNN(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )

        # This made it better
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        sequence_length, batch_size, channels, height, width = x.shape
        features = []
        for t in range(sequence_length):
            image = x[t]
            feature = self.feature_extractor(image)  # batch_size, 2048, 1, 1
            feature = feature.view(batch_size, -1)
            features.append(feature)
        features = torch.stack(features, dim=0)
        rnn_out, _ = self.rnn(features)

        # No relu?

        output = self.fc(self.relu(rnn_out.reshape(sequence_length * batch_size, -1)))
        return output, rnn_out


def collate_fn(batch):
    sequences, labels, same_pairs = zip(*batch)
    padded_sequences = pad_sequence(
        [torch.stack(seq) for seq in sequences], batch_first=True
    )
    stacked_labels = [
        torch.tensor(label) if label is not None else torch.tensor(0)
        for label_seq in labels
        for label in label_seq
    ]
    same_pairs_tensor = torch.tensor(same_pairs, dtype=torch.float32)
    return padded_sequences, stacked_labels, same_pairs_tensor


if __name__ == "__main__":
    set_seed()
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "hidden_size": 512,
            "output_size": 1,
            "num_layers": 3,
            "batch_size": 5,
            "learning_rate": 0.001,
            "num_epochs": 5,
            "optimizer": "SGD",
            "weight_decay": 1e-5,
            "momentum": 0.9,
        },
    )

    config = wandb.config

    # TODO: Change to vary noise
    configs_tasks = TASK_CONFIG_1

    train_task_config = configs_tasks[0]
    IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN = generate_test_transforms(train_task_config)
    num_scramble = 3
    train_dataset = COCO_RNN(
        "/Users/muriellemardenli/Desktop/mainn/ResearchProjects/SNAIL/shared1000/",
        transforms=IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN,
        same_pair_probability=0.5,
        same_not_rand=True,
        idx_ref=4,
        idx_other=5,
        num_scrambled=num_scramble,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=5,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    hidden_size = 512
    output_size = 1
    model = ResNetWithRNN(
        hidden_size=hidden_size, output_size=output_size, num_layers=3
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9
    )
    num_epochs = 5
    all_losses = []
    all_seqs = []
    all_same_pairs = []

    total_iterations = 0

    for epoch in range(num_epochs):
        epoch_losses = []
        print(f"Epoch {epoch + 1}/{num_epochs}")

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
            print(f"  Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

            all_seqs.append(outputs.detach().cpu())
            epoch_losses.append(loss.item())
            all_same_pairs.append(same_pair)

            torch_seq = torch.stack(all_seqs, dim=0)
            iteration = len(all_seqs) - 1
            seq = torch_seq[iteration].reshape(num_seq, batch_size, -1)

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

            wandb.log({"batch_loss": loss.item()})

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        all_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

    torch.save({"losses": all_losses}, "training_data.pth")
    wandb.save("model.pth")

    plt.plot(all_losses)
    plt.ylabel("Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.show()
