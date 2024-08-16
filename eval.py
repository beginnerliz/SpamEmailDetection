# Creator Cui Liz
# Time 28/06/2024 12:01


from Dataset.EmailDataProcess import CacheDataset, DEVICE, collate_fn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from Configs import *
from datetime import datetime
from tqdm import tqdm


def eval(model, return_f1=False, dataset=None):
    model.eval()

    if dataset is None:
        dataset = CacheDataset(["/root/autodl-tmp/test_chunk_12585.pth"], 12585, 0)

    dataloader = DataLoader(dataset, batch_size=100, collate_fn=collate_fn, shuffle=True, drop_last=False)

    all_targets = []
    all_preds = []
    loss_sum = 0
    loss_func = nn.L1Loss()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Testing {model.__class__.__name__}")

        for i, (sentence_embeds, targets) in enumerate(pbar):
            outputs = model(sentence_embeds)  # shape of (B, 1) in range [0, 1]
            pred_label = (outputs >= 0.5).long()

            loss_sum += loss_func(outputs, targets).item()

            all_targets.append(targets)
            all_preds.append(pred_label)

    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    # Print and log results
    report_msg = (
        f"{model.__class__.__name__} : "
        f"Accuracy={accuracy * 100:.4f}%, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1 Score={f1:.4f},"
        f"L1Loss={loss_sum / len(dataset):.4f}"
    )
    print(report_msg)

    model.train()

    if return_f1:
        return report_msg, f1
    else:
        return report_msg


if __name__ == "__main__":
    embed_dim = 768

    # use_model = "trans"
    # use_model = "cnn"
    use_model = "rnn"

    if use_model == "trans":
        from Model.Trans_Classifier import TransformerClassifier

        model_path = "/root/autodl-tmp/20240731-193528/Epoch_29.pth"
        model = TransformerClassifier(768).to(DEVICE)

    elif use_model == "cnn":
        from Model.Res_Classifier import ResClassifier

        model_path = "/root/autodl-tmp/CNN_20240801-095358/Epoch_29.pth"
        model = ResClassifier(embed_dim).to(DEVICE)

    elif use_model == "rnn":
        from Model.RNN_Classifier import RNNClassifier

        model_path = "/root/autodl-tmp/RNN_20240801-180848/Epoch_29.pth"
        model = RNNClassifier(embed_dim).to(DEVICE)

    # Cnn model eval
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    report_msg = eval(model)

    with open("Eval_Results.txt", "a") as out_file:
        out_file.write(datetime.now().strftime('%Y%m%d-%H%M%S') + " : " + report_msg + "\n")




