from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class OracleLinearClassifier(nn.Module):
    def __init__(self, n_features, coefficients):
        super().__init__()
        w = torch.zeros(n_features, dtype=torch.float32)
        c = torch.as_tensor(coefficients, dtype=torch.float32)
        w[:len(c)] = c
        self.register_buffer("weight", w)

    def forward(self, x):
        z = x @ self.weight
        return torch.stack((-z, z), dim=-1)


class MLPClassifier(nn.Module):
    def __init__(self, n_features, hidden_dims=(32,16)):
        super().__init__()
        dims = [n_features, *hidden_dims, 2]
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_mlp(model, x_train, y_train, x_val, y_val, *,
              epochs, batch_size, learning_rate, seed, device="cpu"):
    seed_everything(seed)
    model.to(device)
    ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb)
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(x_val, device=device)
            yv = torch.tensor(y_val, device=device)
            acc = float((model(xv).argmax(1) == yv).float().mean())
        history.append({"epoch": epoch+1, "train_loss": total/len(ds), "val_accuracy": acc})
    return history
