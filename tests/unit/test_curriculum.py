import pytest
import torch
import torch.utils.data
from model.curriculum import (
    CurriculumScheduler,
    CurriculumTrainer,
    EASY_ACTIVITIES,
    MEDIUM_ACTIVITIES,
    HARD_ACTIVITIES,
    PAMAP2_EASY,
    PAMAP2_MEDIUM,
    PAMAP2_HARD,
)


class TestCurriculumScheduler:

    def test_ucihar_phases(self):
        sched = CurriculumScheduler("ucihar", total_epochs=90, num_phases=3)
        assert sched.phases == [EASY_ACTIVITIES, MEDIUM_ACTIVITIES, HARD_ACTIVITIES]

    def test_pamap2_phases(self):
        sched = CurriculumScheduler("pamap2", total_epochs=90, num_phases=3)
        assert sched.phases == [PAMAP2_EASY, PAMAP2_MEDIUM, PAMAP2_HARD]

    def test_uci_har_variant_names(self):
        for name in ("uci-har", "uci_har", "UCIHAR"):
            sched = CurriculumScheduler(name, total_epochs=90, num_phases=3)
            assert sched.phases == [EASY_ACTIVITIES, MEDIUM_ACTIVITIES, HARD_ACTIVITIES]

    def test_get_phase_epoch_0(self):
        sched = CurriculumScheduler("ucihar", total_epochs=90, num_phases=3)
        assert sched.get_phase(0) == 0

    def test_get_phase_last_epoch(self):
        sched = CurriculumScheduler("ucihar", total_epochs=90, num_phases=3)
        assert sched.get_phase(89) == 2

    def test_get_phase_clamped(self):
        sched = CurriculumScheduler("ucihar", total_epochs=90, num_phases=3)
        assert sched.get_phase(200) == 2

    def test_get_active_classes_phase_0(self):
        sched = CurriculumScheduler("ucihar", total_epochs=90, num_phases=3)
        classes = sched.get_active_classes(0)
        assert set(classes) == set(EASY_ACTIVITIES)

    def test_get_active_classes_phase_1(self):
        sched = CurriculumScheduler("ucihar", total_epochs=90, num_phases=3)
        classes = sched.get_active_classes(30)
        assert set(classes) == set(EASY_ACTIVITIES + MEDIUM_ACTIVITIES)

    def test_get_active_classes_all(self):
        sched = CurriculumScheduler("ucihar", total_epochs=90, num_phases=3)
        classes = sched.get_active_classes(89)
        assert set(classes) == set(EASY_ACTIVITIES + MEDIUM_ACTIVITIES + HARD_ACTIVITIES)

    def test_active_classes_sorted(self):
        sched = CurriculumScheduler("ucihar", total_epochs=90, num_phases=3)
        classes = sched.get_active_classes(89)
        assert classes == sorted(classes)


class TestCurriculumTrainer:

    @pytest.fixture
    def tiny_datasets(self):
        """Create tiny train/test datasets with labels from EASY activities."""
        X_train = torch.randn(32, 128, 6)
        y_train = torch.tensor([3, 4, 5] * 10 + [3, 4], dtype=torch.long)
        X_test = torch.randn(16, 128, 6)
        y_test = torch.tensor([3, 4, 5, 3] * 4, dtype=torch.long)
        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        test_ds = torch.utils.data.TensorDataset(X_test, y_test)
        return train_ds, test_ds

    def test_filter_dataset(self, model, tiny_datasets, device):
        train_ds, test_ds = tiny_datasets
        sched = CurriculumScheduler("ucihar", total_epochs=6, num_phases=3)
        trainer = CurriculumTrainer(model, train_ds, test_ds, device, sched, batch_size=8)
        filtered = trainer._filter_dataset(train_ds, [3, 4])
        # Only labels 3 and 4 should remain
        for i in range(len(filtered)):
            _, label = filtered[i]
            assert label.item() in [3, 4]

    def test_train_returns_model_and_history(self, model, tiny_datasets, device):
        train_ds, test_ds = tiny_datasets
        sched = CurriculumScheduler("ucihar", total_epochs=3, num_phases=3)
        trainer = CurriculumTrainer(model, train_ds, test_ds, device, sched, batch_size=8)
        trained_model, history = trainer.train(epochs=3)
        assert trained_model is not None
        assert "train_loss" in history
        assert "test_acc" in history
        assert "phase" in history
        assert "active_classes" in history
        assert len(history["train_loss"]) == 3

    def test_train_empty_filtered_skips(self, model, device):
        """If no samples match active classes, epoch is skipped gracefully."""
        X = torch.randn(8, 128, 6)
        y = torch.full((8,), 0, dtype=torch.long)  # All class 0
        train_ds = torch.utils.data.TensorDataset(X, y)
        test_ds = torch.utils.data.TensorDataset(X, y)
        # EASY = [3,4,5] so class 0 won't match phase 0
        sched = CurriculumScheduler("ucihar", total_epochs=3, num_phases=3)
        trainer = CurriculumTrainer(model, train_ds, test_ds, device, sched, batch_size=8)
        _, history = trainer.train(epochs=1)
        assert history["train_loss"][0] == 0.0

    def test_history_phases_increase(self, model, tiny_datasets, device):
        train_ds, test_ds = tiny_datasets
        sched = CurriculumScheduler("ucihar", total_epochs=6, num_phases=3)
        trainer = CurriculumTrainer(model, train_ds, test_ds, device, sched, batch_size=8)
        _, history = trainer.train(epochs=6)
        # Phases should include at least phase 0 and eventually higher
        assert history["phase"][0] == 0
        assert history["phase"][-1] >= 1
