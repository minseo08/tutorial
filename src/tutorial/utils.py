import random
import hashlib
import torch
import torch.nn as nn
import numpy as np
from petname import english
from dotmap import DotMap
from loguru import logger

def set_random_seed(seed: int) -> None:
    """Fix random seed for reproducibility."""
    logger.info(f"Setting random seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loss_function(metric: str) -> nn.Module:
    """Return the corresponding loss function based on metric."""
    loss_functions = {"rmse": nn.MSELoss(), "mae": nn.L1Loss()}
    if metric not in loss_functions:
        raise ValueError("Unsupported metric. Choose from ['rmse', 'mae'].")
    return loss_functions[metric]

class RunNameGenerator:
    """Singleton class to generate unique names based on hyperparameter hashes."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Ensures only one instance of RunNameGenerator exists."""
        if not cls._instance:
            cls._instance = super(RunNameGenerator, cls).__new__(cls)
        return cls._instance

    def generate_name(self, hyperparams: DotMap) -> str:
        """Generates a unique name based on a hash derived from hyperparameters."""

        def _generate_petname() -> str:
            """Generates a random adjective-name combination."""
            adjective = random.choice(english.adjectives)
            name = random.choice(english.names)
            return f"{adjective}-{name}"

        # Convert hyperparameters to a sorted string representation
        hparam_str = "_".join(
            [f"{k}={v}" for k, v in sorted(hyperparams.items(), key=lambda x: x[0])]
        )

        # Generate a hash integer from the hyperparameter string
        hash_int = int(hashlib.sha256(hparam_str.encode()).hexdigest(), 16)

        # Preserve the current random state to avoid affecting global randomness
        current_state = random.getstate()

        # Seed the random generator with the hash value
        random.seed(hash_int)
        words = _generate_petname()
        number = random.randint(0, 999)

        # Restore the original random state
        random.setstate(current_state)

        return f"{words}-{number}"

