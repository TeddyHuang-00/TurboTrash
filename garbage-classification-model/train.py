import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dataset import get_datasets
from model import MinResNet
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

MULTI_GPU = False
device_ids = None
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    device_ids = list(range(torch.cuda.device_count()))
    MULTI_GPU = True


def train(
    model: nn.Module,
    train: DataLoader,
    test: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    warmup_epochs: int = 3,
    epochs: int = 10,
    patience: int = 5,
):
    if MULTI_GPU:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    # Train 30 epochs to warm up
    for epoch in range(warmup_epochs):
        model.train()
        train_loss = []
        for x, y in tqdm(train):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"[{epoch+1}/{warmup_epochs}] Warming up loss: {np.mean(train_loss):.3f}")

    best_score = np.inf
    count = patience

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for x, y in tqdm(train):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss = np.mean(train_loss)

        model.eval()
        with torch.no_grad():
            test_loss = []
            for x, y in test:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = F.cross_entropy(y_pred, y)
                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)

        print(
            f"[{epoch+1}/{epochs}] Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}"
        )
        score = 0.9 * test_loss + 0.1 * train_loss

        if score < best_score:
            delta = best_score - score
            best_score = score
            count = patience
            torch.save(
                model.state_dict() if not MULTI_GPU else model.module.state_dict(),
                "best_model.pth",
            )
            print(f"Weighted loss decreased: {delta}")
        else:
            count -= 1
            if count == 0:
                print("Early stopped")
                break
            print(f"Early stopping count: {count}")

    if MULTI_GPU:
        model = model.module
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    return model


def eval(model: nn.Module, test: DataLoader, device: torch.device):
    model.load_state_dict(
        torch.load("best_model.pth", weights_only=True, map_location="cpu")
    )
    model.half()
    model.to(device)
    torch.save(model.state_dict(), "best_model_half.pth")

    true = []
    pred = []

    model.eval()
    with torch.no_grad():
        running_loss = []
        for x, y in test:
            x, y = x.to(device), y.to(device)
            y_pred = model(x.half())
            loss = F.cross_entropy(y_pred, y)
            running_loss.append(loss.item())
            true.extend(y.cpu().numpy())
            pred.extend(y_pred.argmax(1).cpu().numpy())
        score = np.mean(running_loss)
    print(f"Validation loss: {score}")

    return true, pred


if __name__ == "__main__":
    train_data, test_data = get_datasets("datasets/GarbageImage12Dataset")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False, num_workers=32)
    model = MinResNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    model = train(
        model,
        train_loader,
        test_loader,
        optimizer,
        device,
        warmup_epochs=5,
        epochs=100,
        patience=10,
    )

    true, pred = eval(model, test_loader, device)

    result = classification_report(true, pred, output_dict=True)
    accuracy = int(str(result["accuracy"])[2:4])  # type: ignore
    if accuracy >= 75:
        now = datetime.now().strftime("%H%M")[:-1] + "0"
        subdir = Path("checkpoints") / f"{now}-{accuracy}"
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "summary.txt").write_text(classification_report(true, pred))  # type: ignore
        subprocess.run(
            ["cp", "best_model.pth", "best_model_half.pth", subdir], check=True
        )
