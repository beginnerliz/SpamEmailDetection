# Creator Cui Liz
# Time 28/06/2024 12:01

from Dataset.EmailDataProcess import CacheDataset, DEVICE, collate_fn
from Model.Trans_Classifier import TransformerClassifier
from Model.Res_Classifier import ResClassifier
from Model.RNN_Classifier import RNNClassifier
from eval import eval

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from Configs import *


def train(model_class, gpt_load_percentage):
    data_path = "/root/autodl-tmp"
    chunk_paths = sorted([f"{data_path}/train_chunk_{i}_15329.pth" for i in [0]])
    dataset = CacheDataset(chunk_paths, 10000, gpt_load_percentage, load_all=True)
    test_set = CacheDataset(["/root/autodl-tmp/train_chunk_8_15329.pth"], 15329, 0, load_all=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=False)

    # Model
    embed_dim = 768
    model = model_class(embed_dim).to(DEVICE)
    model.train()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        **lr_schedule_args
    )

    # Loss function
    loss = nn.BCELoss()

    # Log
    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs(f"/root/autodl-tmp/{model_class.__name__}_{time_now}")
    run_dir = f"/root/tf-logs/{model_class.__name__}_{time_now}"
    writer_train = SummaryWriter(log_dir=run_dir)

    step = 0

    # Train loop
    current_best_f1 = 0
    current_best_report = ""
    # torch.set_grad_enabled(True)
    for epoch in range(n_epochs):
        if model_class == TransformerClassifier:
            if epoch < 15:
                model.onlyCNN()
            else:
                model.enableAll()
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{n_epochs}")
        model_loss = 0.0

        for i, (sentence_embeds, targets) in enumerate(pbar):
            optimizer.zero_grad()
            outputs = model(sentence_embeds)
            result_loss = loss(outputs, targets)
            result_loss.backward()
            optimizer.step()

            if model_loss == 0:
                model_loss = result_loss.item()
            else:
                model_loss = (1 - moving_avg_ratio) * model_loss + moving_avg_ratio * result_loss.item()
            scheduler.step(model_loss)

            step += 1

            pbar.set_postfix_str(f"loss: {result_loss.item():.5e}, lr: {optimizer.param_groups[0]['lr']}")
            if step % log_interval == 0:
                writer_train.add_scalar('Training Loss', model_loss, step)
                writer_train.add_scalar('Training LR', optimizer.param_groups[0]['lr'], step)

        report, f1 = eval(model, True, test_set)
        writer_train.add_scalar("Testing f1", f1, step)
        torch.save(model.state_dict(), f"/root/autodl-tmp/{model_class.__name__}_{time_now}/last.pth")
        if f1 > current_best_f1:
            current_best_f1 = f1
            current_best_report = report
            torch.save(model.state_dict(), f"/root/autodl-tmp/{model_class.__name__}_{time_now}/best.pth")

    writer_train.close()
    del (dataset)
    del (test_set)
    return model, current_best_report


if __name__ == "__main__":
    trained_model, best_report = train(TransformerClassifier, 0.0)
    # trained_model, best_report = train(RNNClassifier, 0.0)
    # trained_model, best_report = train(CNNClassifier, 0.0)

    # with open("Eval_Results.txt", "a") as out_file:
    #     out_file.write("\t" + best_report + "\n")













