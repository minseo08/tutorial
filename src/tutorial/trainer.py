import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from dotmap import DotMap
from tqdm import tqdm
from tutorial.dataset import DataLoader
from tutorial.evaluator import Evaluator
from tutorial import utils
from tutorial.monitor import Monitor


class Trainer:
    """Handles training of model for rating prediction."""
    def __init__(
        self,
        model: nn.Module,
        hyperparams: DotMap,
        evaluator: Evaluator,
        device: torch.device
    ) -> None:
        """Initializes the Trainer class.

        Args:
            model (nn.Module): Latent Factor Model (LFM) to train.
            hyperparams (DotMap): Hyperparameters for training.
            evaluator (Evaluator): Evaluator instance for model evaluation.
            device (torch.device): Device to run training on (CPU or CUDA).
        """
        # Device & Metric Configuration
        self.device = device
        self.metric = hyperparams.metric.lower()
        self.criterion = utils.get_loss_function(self.metric)

        # Training Configuration
        self.seed = hyperparams.seed
        utils.set_random_seed(self.seed)
        self.num_epochs = hyperparams.num_epochs

        # Model & Optimizer Setup
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=hyperparams.lr,
            weight_decay=hyperparams.weight_decay
        )

        # Evaluation & Monitoring
        self.evaluator = evaluator
        self.monitor = Monitor()

        # Best Model Tracking
        self.best_model_state = None
        self.best_valid_score = float("inf")
        self.save_path = self._generate_save_path(hyperparams)
    
    def _generate_save_path(self, hyperparams: DotMap) -> str:
        """Generates a file path for saving the best model.

        Args:
            model (LFM): LFM model instance.
            hyperparams (DotMap): Hyperparameters used for training.

        Returns:
            str: Path to save the model.
        """
        rng = utils.RunNameGenerator()
        file_name = f"{rng.generate_name(hyperparams)}.pth"
        save_dir = "snapshots"
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, file_name)

    def _save_best_model(self, valid_score: float) -> None:
        """Saves the best model state based on validation score.

        Args:
            valid_score (float): Validation score of the current epoch.
        """
        if valid_score < self.best_valid_score: # lower is better for RMSE/MAE
            self.best_valid_score = valid_score
            self.best_model_state = {
                k: v.clone().detach() for k, v in self.model.state_dict().items()
            }
            torch.save(self.best_model_state, self.save_path)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Runs a single training epoch.

        Returns:
            float: Training score (metric value) for the epoch.
        """
        self.model.train()
        total_loss, total_samples = 0.0, 0
        batch_bar = tqdm(
            train_loader,
            desc=f"Training {self.metric.upper()}: 0.0000",
            leave=False
        )

        for batch in batch_bar:
            users, items, ratings = tuple(t.to(self.device) for t in batch)
            loss = self.criterion(self.model(users, items), ratings)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = users.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            avg_score = self.evaluator.compute_metric(total_loss, total_samples)
            batch_bar.set_description(
                f"Training {self.metric.upper()}: {avg_score:.4f}"
            )

        batch_bar.close()
        return self.evaluator.compute_metric(total_loss, total_samples)
    
    def train(self, train_loader: DataLoader, valid_loader: DataLoader) -> Evaluator:
        """Trains the model and validates after each epoch."""
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        epoch_bar = tqdm(
            range(self.num_epochs),
            desc="Epoch Progress",
            leave=False
        )

        for epoch in epoch_bar:
            train_score = self._train_epoch(train_loader)
            valid_score = self.evaluator.evaluate(self.model, valid_loader)

            self._save_best_model(valid_score)

            epoch_bar.set_description(
                f"Epoch {epoch + 1} - "
                f"Train {self.metric.upper()}: {train_score:.4f}, "
                f"Valid {self.metric.upper()}: {valid_score:.4f}"
            )

            self.monitor.log({
                "epoch": epoch + 1,
                f"train_{self.metric}": train_score,
                f"valid_{self.metric}": valid_score
            })

        epoch_bar.close()
        elapsed_time = epoch_bar.format_dict["elapsed"]
        logger.info(f"Training complete! Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"Best Valid {self.metric.upper()}: {self.best_valid_score:.4f}")

        # Return the best model over iterations in terms of validation
        best_model = copy.deepcopy(self.model)
        best_model.load_state_dict(self.best_model_state)
        return best_model
