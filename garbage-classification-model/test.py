from pathlib import Path

import numpy as np
import torch
from dataset import get_datasets
from model import MinResNet
from sklearn.metrics import classification_report
from torch.nn import functional as F
from torch.utils.data import DataLoader


def test(test: DataLoader, device: torch.device):
    CHECKPOINT_DIR = Path("checkpoints")
    models = []
    for checkpoint in CHECKPOINT_DIR.iterdir():
        _model = MinResNet().half()
        _model.load_state_dict(
            torch.load(checkpoint / "best_model_half.pth", weights_only=True)
        )
        _model.to(device)
        _model.eval()
        models.append(_model)

    true = []
    pred = []

    with torch.no_grad():
        running_loss = []
        for x, y in test:
            x, y = x.to(device), y.to(device)
            y_pred = torch.stack([model(x.half()) for model in models]).mean(0)
            loss = F.cross_entropy(y_pred, y)
            running_loss.append(loss.item())
            true.extend(y.cpu().numpy())
            pred.extend(y_pred.argmax(1).cpu().numpy())
        score = np.mean(running_loss)
    print(f"Validation loss: {score}")

    return true, pred


if __name__ == "__main__":
    train_data, test_data = get_datasets("datasets/GarbageImage12Dataset")
    print(train_data.classes, test_data.classes)

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true, pred = test(test_loader, device)

    print(classification_report(true, pred))
