import argparse
import os
import torch
from loguru import logger
from tutorial.dataset import MovieLensDataset, SplitDataLoader
from tutorial.evaluator import Evaluator
from tutorial.trainer import Trainer
from tutorial.model import LFM
from tutorial.monitor import Monitor

# Disable logging if the LOG_DISABLE environment variable is set
if os.getenv("LOG_DISABLE") == "1":
    logger.remove()

def parse_args():
    """Parse command-line arguments for ML training."""
    parser = argparse.ArgumentParser(description="Train ML models for rating prediction")

    # General
    parser.add_argument("--project", type=str, default="tutorial", help="Project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model
    parser.add_argument("--model", type=str, default="lmf", help="Model type")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Embedding size")

    # Training
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="L2 regularization")
    parser.add_argument("--num_epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")

    # Data
    parser.add_argument("--data_path", type=str, default="./data/ML100K.csv",
                        help="Dataset path")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train set ratio")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="Validation ratio")

    # Evaluation
    parser.add_argument("--metric", type=str, default="rmse",
                        choices=["rmse", "mae"], help="Eval metric")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")

    return parser.parse_args()

def main():
    """
    Train ML models for rating prediction
    """
    monitor = Monitor(parse_args())
    hyperparams = monitor.get_hyperparams()

    # Log hyperparameters
    logger.info("Using Hyperparameters:")
    for key, value in hyperparams.items():
        logger.info(f"  {key}: {value}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize data loader with training, validation, and test splits
    split = SplitDataLoader(
        dataset=MovieLensDataset(hyperparams.data_path),
        train_ratio=hyperparams.train_ratio,
        valid_ratio=hyperparams.valid_ratio,
        batch_size=hyperparams.batch_size,
        seed=hyperparams.seed
    )

    # Initialize Latent Factor Model (LFM)
    model = LFM(
        num_users=split.num_users,
        num_items=split.num_items,
        embedding_dim=hyperparams.embedding_dim
    )

    # Initialize evaluator
    evaluator = Evaluator(
        metric=hyperparams.metric,
        device=device
    )

    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        hyperparams=hyperparams,
        evaluator=evaluator,
        device=device
    )

    # Train the model using the training set while validating
    model = trainer.train(split.train_loader, split.valid_loader)

    # Evaluate the model for the test set
    test_score = evaluator.evaluate(model, split.test_loader)
    logger.info(f"Final Test {hyperparams.metric.upper()}: {test_score:.4f}")
    monitor.log({f"test_{hyperparams.metric}": test_score})

    # finalize
    monitor.finish()


if __name__ == "__main__":
    main()
