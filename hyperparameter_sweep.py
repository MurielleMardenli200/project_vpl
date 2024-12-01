#!/usr/bin/env python3
from constants.task_type import TASK_CONFIG_1
from constants.wandb_constants import WANDB_RUN_NAME
from generate_base import ResNetWithRNN, collate_fn, set_seed
import torch
import torch.nn as nn
import numpy as np
from itertools import product

import matplotlib

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# import wandb
from database.coco_1 import generate_test_transforms, COCO_RNN

# Wandb login


def train(config):
    # with wandb.init(name=WANDB_RUN_NAME, config=config, mode="offline"):
    # config = wandb.config
    print("training!")

    configs_tasks = TASK_CONFIG_1
    train_task_config = configs_tasks[0]
    IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN = generate_test_transforms(train_task_config)
    print("aparture")
    print(IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN)
    train_dataset = COCO_RNN(
        "shared1000/",
        transforms=IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN,
        same_pair_probability=0.5,
        same_not_rand=True,
        idx_ref=4,
        idx_other=5,
        num_scrambled=3,
    )
    print("rnn")
    print(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print("data loaded")

    model = ResNetWithRNN(
        hidden_size=config["hidden_size"],
        output_size=1,
        num_layers=config["num_layers"],
    )
    print("model resnet rnn")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        momentum=0.9,
    )

    print("optimizer")

    all_seqs = []
    total_iterations = 0
    total_loss = []

    for epoch in range(config["num_epochs"]):
        epoch_losses = []
        for batch_idx, (sequences, labels, same_pairs) in enumerate(train_loader):
            print(f"batch id {batch_idx}")
            # total_iterations += 1
            # same_pair = same_pairs[1]
            # num_seq = sequences.shape[1]
            # batch_size = sequences.shape[0]

            # print(f"config {config}")
            # optimizer.zero_grad()
            # outputs, seqs = model(sequences.permute(1, 0, 2, 3, 4))
            # print("after sqs")

            # outputs_loss = outputs.reshape(num_seq, batch_size, -1)[-1]

            # loss = criterion(outputs_loss.squeeze(), same_pairs)
            # print("criterion")

            # loss.backward()
            # print("backward")

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()
            # print("optimizer step")

            # epoch_losses.append(loss.item())
            # all_seqs.append(outputs.detach().cpu())

            # torch_seq = torch.stack(all_seqs, dim=0)
            # print("stack")

            # iteration = len(all_seqs) - 1
            # seq = torch_seq[iteration].reshape(num_seq, batch_size, -1)

            # # wandb.log({"batch_loss": loss.item()})

            # plt.figure(figsize=(10, 6))
            # for i in range(sequences.shape[0]):
            #     plt.plot(
            #         seq[:, i].detach().cpu().numpy(),
            #         label=f"batch index {i}",
            #         marker="o",
            #     )
            # plt.legend()

            # total_loss.append(loss.item())
            # plt.plot(list(range(total_iterations)), total_loss)

            # plt.title(f"batch_loss")
            # plt.ylabel("Loss value")
            # plt.xlabel("Sequences")
            # hidden_size = config["hidden_size"]
            # batch_s = config["batch_size"]
            # learning_rate = config["learning_rate"]
            # num_layers = config["num_layers"]
            # plt.savefig(
            #     f"loss_{hidden_size}_{batch_s}_{learning_rate}_{num_layers}.png"
            # )
            # plt.grid(True)
            # plt.close()
            # print(f"output for iteration {total_iterations}")

            # # if batch_idx % 5 == 0:
            # #     wandb.log({"RNN_output_plot": wandb.Image("temp_plot.png")})

            # current_lr = optimizer.param_groups[0]["lr"]
            # wandb.log({"learning_rate": current_lr})

        # epoch_loss = np.mean(epoch_losses)
        # wandb.log({"epoch_loss": epoch_loss})
    # torch.save(model.state_dict(), "model.pth")
    # wandb.save("model.pth")


# # sweep_config = {
# #     "optimizer": "SGD",
# #     "method": "grid",
# #     "metric": {"name": "epoch_loss", "goal": "minimize"},
# #     "parameters": {
# #         "hidden_size": {"values": [256, 512, 1024]},
# #         "batch_size": {"values": [5, 10, 20]},
# #         "learning_rate": {"values": [0.001, 0.005, 0.01]},
# #         "num_layers": {"values": [3, 4, 5]},
# #         "weight_decay": {"values": [1e-5]},
# #         "num_epochs": {"values": [3]},
# #     },
# # }


def get_hyperparameter_combinations(sweep_config):
    keys, values = zip(*sweep_config.items())
    for combination in product(*(v["values"] for v in values)):
        yield dict(zip(keys, combination))


sweep_config = {
    "hidden_size": {"values": [256, 512, 1024]},
    "batch_size": {"values": [5, 10, 20]},
    "learning_rate": {"values": [0.001, 0.005, 0.01]},
    "num_layers": {"values": [3, 4, 5]},
    "weight_decay": {"values": [1e-5]},
    "num_epochs": {"values": [3]},
}

if __name__ == "__main__":
    print("gonna training!")
    set_seed()
    # sweep_id = wandb.sweep(sweep_config, project="HyperparamSearch_compute")
    # wandb.agent(sweep_id, train)

    for config in get_hyperparameter_combinations(sweep_config):
        train(config)
