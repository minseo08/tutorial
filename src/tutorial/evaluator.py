import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
from tutorial import utils


class Evaluator:
    """Handles evaluation of the LFM model."""

    def __init__(
        self,
        metric: str,
        device: torch.device
    ):
        self.metric = metric
        self.criterion = utils.get_loss_function(self.metric)
        self.device = device

    def compute_metric(
        self,
        total_loss: float,
        total_samples: int
    ) -> float:
        """Compute RMSE or MAE metric for the epoch."""
        if self.metric not in {"rmse", "mae"}:
            raise ValueError(f"Unsupported metric: {self.metric}. Use 'rmse' or 'mae'.")

        avg_loss = total_loss / total_samples
        return math.sqrt(avg_loss) if self.metric == "rmse" else avg_loss

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> float:
        """Evaluate the model for the given dataset."""
        model.eval()
        total_loss, total_samples = 0.0, 0
        batch_bar = tqdm(
            data_loader,
            desc=f"Evaluating {self.metric.upper()}: 0.0000",
            leave=False
        )

        with torch.no_grad():
            for batch in batch_bar:
                users, items, ratings = tuple(t.to(self.device) for t in batch)
                loss = self.criterion(model(users, items), ratings)

                batch_size = users.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                avg_score = self.compute_metric(total_loss, total_samples)

                batch_bar.set_description(
                    f"Evaluating {self.metric.upper()}: {avg_score:.4f}"
                )

        batch_bar.close()
        return self.compute_metric(total_loss, total_samples)
