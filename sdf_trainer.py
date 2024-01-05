import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import torch
from torch import no_grad
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from loss import mape_loss
from sdf_network import SDFNetwork


class SDFTrainer:
    def __init__(self, model: SDFNetwork, train_data_loader: DataLoader, validation_data_loader: DataLoader,
                 learning_rate: float, epochs: int, output_path: Path):
        self._model: SDFNetwork = model
        self._train_data_loader: DataLoader = train_data_loader
        self._validation_data_loader: DataLoader = validation_data_loader
        self._learning_rate: float = learning_rate
        self._epochs: int = epochs
        self._criterion = mape_loss

        self._optimizer: Adam = Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=self._learning_rate, betas=(0.9, 0.99), eps=1e-15)

        self._lr_scheduler: StepLR = StepLR(self._optimizer, step_size=10, gamma=0.1)

        self._ema: ExponentialMovingAverage = ExponentialMovingAverage(self._model.parameters(), decay=0.95)

        self._scaler: GradScaler = GradScaler(enabled=True)

        self._epoch: int = 0
        self._stats: Dict[str, Any] = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        self._output_path: Path = output_path
        self._checkpoints_path: Path = self._output_path / "checkpoints"
        self._best_path: Path = self._output_path / "best.pth"

        os.makedirs(self._output_path, exist_ok=True)
        os.makedirs(self._checkpoints_path, exist_ok=True)

        self._max_checkpoints: int = 2

        self._device: str = 'cuda'

    def _train_step(self, data):
        return self._criterion(self._model(data["points"][0]), data["sdfs"][0])

    def _eval_step(self, data):
        return self._train_step(data)

    def _test_step(self, data):
        return self._model(data["points"][0])

    def train(self, description: str = ""):
        with tqdm(range(self._epoch + 1, self._epochs + 1)) as pbar:
            current_status = {}

            def set_pbar_status(**kwargs):
                current_status.update(kwargs)
                pbar.set_description(f"Training{' ' if description else ''}{description} - " +
                                     ', '.join(f"{key}={val}" for key, val in current_status.items()))

            def on_train_loss_changed(new_loss: float):
                set_pbar_status(train_loss=f"{new_loss:.4f}")

            for epoch in pbar:
                self._epoch = epoch

                self._train_one_epoch(on_train_loss_changed)
                self._save_checkpoint(full=True, best=False)
                self._evaluate()
                set_pbar_status(val_loss=f"{self._stats['valid_loss'][-1]:.4f}")
                pbar.refresh()
                self._save_checkpoint(full=False, best=True)

    def _prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self._device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self._device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self._device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self._device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self._device, non_blocking=True)
        else:  # is_tensor, or other similar objects that has `to`
            data = data.to(self._device, non_blocking=True)

        return data

    def _train_one_epoch(self, on_loss_changed: Optional[Callable[[float], None]] = None):
        total_loss = 0

        self._model.train()

        step = 0

        for data in self._train_data_loader:
            step += 1

            data = self._prepare_data(data)

            self._optimizer.zero_grad()

            with autocast(enabled=True):
                loss = self._train_step(data)

            self._scaler.scale(loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()

            self._ema.update()

            loss_val = loss.item()
            total_loss += loss_val

            on_loss_changed(loss_val)

        average_loss = total_loss / step
        self._stats["loss"].append(average_loss)

        self._lr_scheduler.step()

    def _evaluate(self):
        total_loss = 0

        self._model.eval()

        with no_grad():
            step = 0
            for data in self._validation_data_loader:
                step += 1

                data = self._prepare_data(data)

                self._ema.store()
                self._ema.copy_to()

                with autocast(enabled=True):
                    loss = self._eval_step(data)

                self._ema.restore()

                loss_val = loss.item()
                total_loss += loss_val

        average_loss = total_loss / step
        self._stats["valid_loss"].append(average_loss)
        self._stats["results"].append(average_loss)

    def _save_checkpoint(self, full=False, best=False):
        state = {
            'epoch': self._stats,
            'stats': self._stats,
        }

        if full:
            state['optimizer'] = self._optimizer.state_dict()
            state['lr_scheduler'] = self._lr_scheduler.state_dict()
            state['scaler'] = self._scaler.state_dict()
            state['ema'] = self._ema.state_dict()

        if best:
            if len(self._stats["results"]) > 0:
                if self._stats["best_result"] is None or self._stats["results"][-1] < self._stats["best_result"]:
                    self._stats["best_result"] = self._stats["results"][-1]

                    self._ema.store()
                    self._ema.copy_to()

                    state['model'] = self._model.state_dict()

                    self._ema.restore()

                    torch.save(state, self._best_path)
        else:
            state['model'] = self._model.state_dict()

            file_path = f"{self._checkpoints_path}/epoch_{self._epoch:03d}.pth"

            self._stats["checkpoints"].append(file_path)

            if len(self._stats["checkpoints"]) > self._max_checkpoints:
                old_ckpt = self._stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)
