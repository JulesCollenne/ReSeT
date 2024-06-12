import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data_isic import DataLoaderISIC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve

from reset import ReSeT

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--train_epochs", type=int, default=150)
args = parser.parse_args()


def main():
    model_name = "CNN"

    dim_input = sum(['feature' in col for col in pd.read_csv(f"features/{model_name}_val.csv").columns])
    num_outputs = 1
    emb_dim = 20
    dim_output = 2
    n_vect = 50

    val_gen = DataLoaderISIC(
        f"features/{model_name}_val.csv",
        "GroundTruth.csv",
        batch_size=args.batch_size,
        n_vect=n_vect
    )

    test_gen = DataLoaderISIC(
        f"features/{model_name}_test.csv",
        "GroundTruth.csv",
        batch_size=args.batch_size,
        n_vect=n_vect
    )

    class_weights = torch.tensor([0.02, 0.98]).cuda()
    criterion = nn.CrossEntropyLoss(class_weights)

    model = ReSeT(dim_input, num_outputs, emb_dim, dim_output)

    model = nn.DataParallel(model)
    model = model.cuda()

    model.load_state_dict(torch.load(f"models/{model_name}/{model_name}.pth"))
    model.eval()

    losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
    base_fpr = np.linspace(0, 1, 101)

    for imgs, lbls in val_gen.test_data():
        imgs = torch.Tensor(imgs).cuda()
        lbls = torch.Tensor(lbls).long().cuda()
        preds = model(imgs)

        zero_rows_mask = torch.all(imgs == 0, dim=2)
        non_zero_rows_mask = ~zero_rows_mask
        preds = preds[non_zero_rows_mask]
        lbls = lbls[non_zero_rows_mask]

        loss = criterion(preds.view(-1, 2), lbls.view(-1))

        losses.append(loss.item())
        total += lbls.view(-1).shape[0]
        correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

        true_labels += lbls.view(-1).cpu().numpy().tolist()
        predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()

    avg_loss, avg_acc = np.mean(losses), correct / total

    auc = roc_auc_score(true_labels, predicted_probs)
    best_bacc = 0
    best_thresh = 0
    thresholds = np.linspace(0, 1, 5)

    for thresh in thresholds:
        balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > thresh).astype(int))
        if balanced_acc > best_bacc:
            best_bacc = balanced_acc
            best_thresh = thresh

    print(
        f"val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f}"
        f" val balanced acc {best_bacc:.3f} best threshold {best_thresh:.3f}")

    losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
    for imgs, lbls in test_gen.test_data():
        imgs = torch.Tensor(imgs).cuda()
        lbls = torch.Tensor(lbls).long().cuda()
        preds = model(imgs)

        zero_rows_mask = torch.all(imgs == 0, dim=2)
        non_zero_rows_mask = ~zero_rows_mask
        preds = preds[non_zero_rows_mask]
        lbls = lbls[non_zero_rows_mask]

        loss = criterion(preds.view(-1, 2), lbls.view(-1))

        losses.append(loss.item())
        total += lbls.view(-1).shape[0]
        correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

        true_labels += lbls.view(-1).cpu().numpy().tolist()
        predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()

    avg_loss, avg_acc = np.mean(losses), correct / total

    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    np.savetxt(f"predictions/{model_name}", np.array(predicted_probs), delimiter=",")
    binary_predictions = (np.array(predicted_probs) > best_thresh).astype(int)

    conf_matrix = confusion_matrix(true_labels, binary_predictions)

    auc = roc_auc_score(true_labels, predicted_probs)
    bacc = balanced_accuracy_score(true_labels, binary_predictions)
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0

    print(
        f"test loss {avg_loss:.3f} test acc {avg_acc:.3f} test AUC {auc:.3f}"
        f" test balanced acc {bacc:.3f}")


if __name__ == "__main__":
    main()
