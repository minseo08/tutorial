import os
import pytest
from dotmap import DotMap
import torch
from tutorial.dataset import MovieLensDataset, SplitDataLoader
from tutorial.model import LFM
from tutorial.trainer import Trainer
from tutorial.evaluator import Evaluator
from tutorial.monitor import Monitor


@pytest.fixture
def split_fixture():
    """Creates and returns a SplitDataLoader instance."""
    args = DotMap({
        "model": "LFM",
        "embedding_dim": 64,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "num_epochs": 3,
        "batch_size": 128,
        "seed": 42,
        "metric": "rmse",
        "data_path": "./data/ML100K.csv",
        "train_ratio": 0.8,
        "valid_ratio": 0.1,
        "wandb": False
    })
    Monitor(args)

    dataset = MovieLensDataset(data_path=args.data_path)
    split = SplitDataLoader(
        dataset, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio,
        batch_size=args.batch_size, seed=args.seed
    )
    return split


@pytest.fixture
def evaluator_fixture():
    """Creates and returns an Evaluator instance."""
    monitor = Monitor()
    hyperparams = monitor.get_hyperparams()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(hyperparams.metric, device)
    return evaluator


@pytest.fixture
def trainer_fixture(split_fixture, evaluator_fixture):
    """Creates and returns a Trainer instance with predefined hyperparameters."""
    monitor = Monitor()
    hyperparams = monitor.get_hyperparams()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LFM(
        num_users=split_fixture.num_users,
        num_items=split_fixture.num_items,
        embedding_dim=hyperparams.embedding_dim
    )

    trainer = Trainer(model, hyperparams, evaluator_fixture, device)
    trainer.split = split_fixture
    return trainer


def test_train_with_movielens(split_fixture, evaluator_fixture, trainer_fixture):
    """Trains the model and ensures RMSE is within an acceptable range."""
    model = trainer_fixture.train(
        split_fixture.train_loader,
        split_fixture.valid_loader
    )
    test_score = evaluator_fixture.evaluate(model, split_fixture.test_loader)

    print(f"Final Test RMSE: {test_score:.4f}")
    assert test_score < 1.0, f"Test RMSE is too high: {test_score:.4f}"


def test_model_save_path(trainer_fixture, split_fixture):
    """Test whether the model is saved correctly and delete it after the test."""
    trainer_fixture.train(
        split_fixture.train_loader,
        split_fixture.valid_loader
    )

    assert os.path.exists(trainer_fixture.save_path)
    assert os.path.getsize(trainer_fixture.save_path) > 0
    os.remove(trainer_fixture.save_path)
    assert not os.path.exists(trainer_fixture.save_path)