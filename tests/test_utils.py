import pytest
import random
import re
import torch
import numpy as np
from dotmap import DotMap
from tutorial import utils
from tutorial.utils import RunNameGenerator

def test_set_random_seed():
    """Test if setting a random seed ensures reproducibility across different RNGs."""
    seed = 42
    utils.set_random_seed(seed)

    assert torch.initial_seed() == seed
    assert np.random.get_state()[1][0] == seed
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

def test_get_loss_function():
    """Verify that the correct loss function is returned and invalid inputs raise an error."""
    mse_loss = utils.get_loss_function("rmse")
    mae_loss = utils.get_loss_function("mae")

    assert isinstance(mse_loss, torch.nn.MSELoss)
    assert isinstance(mae_loss, torch.nn.L1Loss)

    error_msg = "Unsupported metric. Choose from ['rmse', 'mae']."
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        utils.get_loss_function("invalid_metric")

def test_generate_name():
    """Test if the same hyperparameters generate the same name,
    and different hyperparameters generate different names."""

    rng = RunNameGenerator()  # Use the singleton instance

    for _ in range(100):  # Run 100 test iterations
        # Generate random hyperparameters
        hyperparams = DotMap({
            "lr": round(random.uniform(0.001, 0.1), 5),
            "epochs": random.randint(1, 100),
            "batch_size": random.choice([16, 32, 64, 128]),
        })

        name1 = rng.generate_name(hyperparams)
        name2 = rng.generate_name(hyperparams)

        # Ensure the same hyperparameters generate the same name
        assert name1 == name2

        # Modify one hyperparameter slightly
        different_hyperparams = hyperparams.copy()
        different_hyperparams["lr"] += 0.0001

        name3 = rng.generate_name(different_hyperparams)

        # Ensure different hyperparameters generate a different name
        assert name1 != name3