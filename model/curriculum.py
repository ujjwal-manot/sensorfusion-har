import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

EASY_ACTIVITIES = [3, 4, 5]
MEDIUM_ACTIVITIES = [0]
HARD_ACTIVITIES = [1, 2]

PAMAP2_EASY = [0, 1, 2]
PAMAP2_MEDIUM = [3, 6]
PAMAP2_HARD = [4, 5, 7, 8, 9, 10, 11]


class CurriculumScheduler:

    def __init__(self, dataset_name="ucihar", total_epochs=100, num_phases=3):
        self.dataset_name = dataset_name.lower()
        self.total_epochs = total_epochs
        self.num_phases = num_phases

        if self.dataset_name in ("ucihar", "uci-har", "uci_har"):
            self.phases = [EASY_ACTIVITIES, MEDIUM_ACTIVITIES, HARD_ACTIVITIES]
        else:
            self.phases = [PAMAP2_EASY, PAMAP2_MEDIUM, PAMAP2_HARD]

        self.phase_epochs = total_epochs // num_phases

    def get_phase(self, epoch):
        phase = min(epoch // max(self.phase_epochs, 1), self.num_phases - 1)
        return phase

    def get_active_classes(self, epoch):
        phase = self.get_phase(epoch)
        active = []
        for p in range(phase + 1):
            active.extend(self.phases[p])
        return sorted(set(active))


class CurriculumTrainer:

    def __init__(self, model, train_ds, test_ds, device, scheduler, batch_size=64, lr=0.001, entropy_weight=0.01):
        self.model = model.to(device)
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.device = device
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.lr = lr
        self.entropy_weight = entropy_weight

    def _filter_dataset(self, dataset, active_classes):
        indices = []
        for i in range(len(dataset)):
            label = dataset[i][1]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label in active_classes:
                indices.append(i)
        return Subset(dataset, indices)

    def train(self, epochs=100):
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {
            "train_loss": [],
            "test_acc": [],
            "phase": [],
            "active_classes": [],
        }

        for epoch in range(epochs):
            self.model.train()
            active_classes = self.scheduler.get_active_classes(epoch)
            phase = self.scheduler.get_phase(epoch)

            filtered_train = self._filter_dataset(self.train_ds, active_classes)

            if len(filtered_train) == 0:
                history["train_loss"].append(0.0)
                history["test_acc"].append(0.0)
                history["phase"].append(phase)
                history["active_classes"].append(active_classes)
                continue

            train_loader = DataLoader(filtered_train, batch_size=self.batch_size, shuffle=True, drop_last=len(filtered_train) >= self.batch_size)

            total_loss = 0.0
            num_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if self.entropy_weight > 0 and hasattr(self.model, 'forward') and 'return_aux' in self.model.forward.__code__.co_varnames:
                    logits, aux = self.model(batch_x, return_aux=True)
                    loss = criterion(logits, batch_y)
                    loss = loss - self.entropy_weight * aux["attention_entropy"]
                else:
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            lr_scheduler.step()

            avg_loss = total_loss / max(num_batches, 1)

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    preds = self.model(batch_x)
                    correct += (preds.argmax(dim=1) == batch_y).sum().item()
                    total += batch_y.size(0)

            acc = correct / max(total, 1)

            history["train_loss"].append(avg_loss)
            history["test_acc"].append(acc)
            history["phase"].append(phase)
            history["active_classes"].append(active_classes)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Phase: {phase} Classes: {active_classes} Loss: {avg_loss:.4f} Acc: {acc:.4f}")

        return self.model, history
