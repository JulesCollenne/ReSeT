import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data_isic import DataLoaderISIC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from reset import ReSeT

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--train_epochs", type=int, default=200)
parser.add_argument("--resume", type=str, default="")
args = parser.parse_args()


def main():
    model_name = "CNN"

    n_vect = 200
    dim_input = sum(['feature' in col for col in pd.read_csv(f"features/{model_name}_val.csv").columns])
    num_outputs = 1
    emb_dim = 200
    dim_output = 2
    patience = 15

    train_gen = DataLoaderISIC(
        f"features/{model_name}_train.csv",
        "GroundTruth.csv",
        batch_size=args.batch_size,
        n_vect=n_vect
    )

    val_gen = DataLoaderISIC(
        f"features/{model_name}_val.csv",
        "GroundTruth.csv",
        batch_size=args.batch_size,
        n_vect=n_vect
    )

    class_counts = Counter(train_gen.gt['target'])
    total_count = sum(class_counts.values())
    class_weights = {cls: total_count / count for cls, count in class_counts.items()}
    class_weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float).cuda()

    model = ReSeT(dim_input, num_outputs, emb_dim, dim_output)

    model = nn.DataParallel(model)
    model = model.cuda()

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    if not os.path.exists(f"models/{model_name}"):
        os.makedirs(f"models/{model_name}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # class_weights = torch.tensor([0.02, 0.98]).cuda()
    criterion = nn.CrossEntropyLoss(class_weights)
    old_mean = 0
    current_patience = 0

    print("Training:", model_name)
    for epoch in range(args.train_epochs):
        print(f"Epoch: {epoch}")
        model.train()
        losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []

        for imgs, lbls in train_gen.data():
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = model(imgs)

            zero_rows_mask = torch.all(imgs == 0, dim=2)
            non_zero_rows_mask = ~zero_rows_mask
            preds = preds[non_zero_rows_mask]
            lbls = lbls[non_zero_rows_mask]

            loss = criterion(preds.view(-1, 2), lbls.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total += lbls.view(-1).shape[0]
            correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

            true_labels += lbls.view(-1).cpu().numpy().tolist()
            predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()
            print(f"Batch loss: {loss:.3f} correct: {correct / total * 100:.3f}%")

        avg_loss, avg_acc = np.mean(losses), correct / total

        auc = roc_auc_score(true_labels, predicted_probs)
        balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))

        print(
            f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f} train AUC {auc:.3f} train balanced acc {balanced_acc:.3f}")

        model.eval()
        losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
        with torch.no_grad():
            for imgs, lbls in val_gen.data():
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
        balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))
        new_mean = auc

        print(
            f"Epoch {epoch}: val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f} val balanced acc {balanced_acc:.3f}")

        if new_mean >= old_mean:
            torch.save(model.state_dict(), f"models/{model_name}/{model_name}.pth")
            print("Saving model...")
            old_mean = new_mean
            current_patience = 0
        else:
            current_patience += 1

        if current_patience >= patience:
            print("Stopping training.")
            break

            if epoch % 1 == 0:
                print(
                    f"Epoch {epoch}: val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f} "
                    f"val balanced acc {balanced_acc:.3f}")


if __name__ == "__main__":
    main()
