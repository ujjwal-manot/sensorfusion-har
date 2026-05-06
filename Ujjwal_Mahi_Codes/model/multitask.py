import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .reservoir import EchoStateNetwork
from .dsconv import DSConvEncoder
from .attention import PatchMicroAttention
from .binary_head import BinaryClassifier
from .sensorfusion import SpectralGatedFusion, SensorFusionHAR


class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class SubjectLabeledDataset(Dataset):

    def __init__(self, X, y, subjects):
        self.X = X
        self.y = y
        self.subjects = subjects

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subjects[idx]


class MultiTaskHAR(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, num_classes=6, num_subjects=30):
        super().__init__()
        self.reservoir = EchoStateNetwork(
            input_channels, reservoir_size,
            learnable_sr=True, use_diff_states=True, reservoir_dropout=0.1
        )
        self.diff_gate = nn.Parameter(torch.zeros(reservoir_size))
        self.dsconv = DSConvEncoder(in_channels=reservoir_size)
        self.gate = SpectralGatedFusion(reservoir_dim=reservoir_size, dsconv_channels=48, seq_len=32)
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.activity_head = BinaryClassifier(in_features=32, num_classes=num_classes)
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.subject_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_subjects),
        )

    def _merge_reservoir_states(self, h):
        rs = self.reservoir.reservoir_size
        alpha = torch.sigmoid(self.diff_gate)
        return h[:, :, :rs] + alpha * h[:, :, rs:]

    def forward(self, x):
        h = self.reservoir(x)
        h = self._merge_reservoir_states(h)
        h = h.transpose(1, 2)
        dsconv_out = self.dsconv(h)
        x = self.gate(h, dsconv_out)
        features = self.attention(x)
        activity_logits = self.activity_head(features)
        reversed_features = self.grl(features)
        subject_logits = self.subject_head(reversed_features)
        return activity_logits, subject_logits

    def extract_backbone(self):
        backbone = SensorFusionHAR(
            input_channels=self.reservoir.W_in.shape[0],
            reservoir_size=self.reservoir.reservoir_size,
            num_classes=self.activity_head.head.linear.out_features,
        )
        backbone.reservoir.load_state_dict(self.reservoir.state_dict(), strict=False)
        backbone.dsconv.load_state_dict(self.dsconv.state_dict())
        backbone.gate.load_state_dict(self.gate.state_dict(), strict=False)
        backbone.attention.load_state_dict(self.attention.state_dict())
        backbone.classifier.load_state_dict(self.activity_head.state_dict(), strict=False)
        return backbone


def _build_subject_dataset(dataset, root_dir, split):
    import os
    subjects_path = os.path.join(root_dir, split, "subject_{}.txt".format(split))
    if os.path.exists(subjects_path):
        subject_ids = np.loadtxt(subjects_path, dtype=int)
        unique_subjects = sorted(set(subject_ids))
        subject_map = {s: i for i, s in enumerate(unique_subjects)}
        mapped = torch.tensor([subject_map[s] for s in subject_ids], dtype=torch.long)
        return SubjectLabeledDataset(dataset.X, dataset.y, mapped), len(unique_subjects)
    if hasattr(dataset, "subjects"):
        return SubjectLabeledDataset(dataset.X, dataset.y, dataset.subjects), int(dataset.subjects.max().item()) + 1
    dummy = torch.zeros(len(dataset), dtype=torch.long)
    return SubjectLabeledDataset(dataset.X, dataset.y, dummy), 1


def train_multitask(model, train_ds, test_ds, device, epochs=100, batch_size=64, lr=0.001, lambda_schedule="linear"):
    model = model.to(device)
    activity_criterion = nn.CrossEntropyLoss()
    subject_criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        model.train()

        if lambda_schedule == "linear":
            progress = min(epoch / max(epochs // 2, 1), 1.0)
        else:
            progress = 1.0
        model.grl.set_lambda(progress)

        total_act_loss = 0.0
        total_subj_loss = 0.0
        num_batches = 0

        for batch_x, batch_y, batch_subj in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_subj = batch_subj.to(device)

            activity_logits, subject_logits = model(batch_x)

            act_loss = activity_criterion(activity_logits, batch_y)
            subj_loss = subject_criterion(subject_logits, batch_subj)
            loss = act_loss + progress * subj_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_act_loss += act_loss.item()
            total_subj_loss += subj_loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
                for batch in test_loader:
                    bx = batch[0].to(device)
                    by = batch[1].to(device)
                    if isinstance(model, MultiTaskHAR):
                        preds, _ = model(bx)
                    else:
                        preds = model(bx)
                    correct += (preds.argmax(dim=1) == by).sum().item()
                    total += by.size(0)

            avg_act = total_act_loss / max(num_batches, 1)
            avg_subj = total_subj_loss / max(num_batches, 1)
            acc = correct / max(total, 1)
            print(f"Epoch [{epoch + 1}/{epochs}] ActLoss: {avg_act:.4f} SubjLoss: {avg_subj:.4f} Lambda: {progress:.2f} Acc: {acc:.4f}")

    return model
