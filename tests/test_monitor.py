import pytest
from tutorial.monitor import Monitor

class DummyArgs:
    def __init__(self):
        self.model = "TestModel"
        self.embedding_dim = 128
        self.lr = 0.001
        self.weight_decay = 0.0001
        self.num_epochs = 10
        self.batch_size = 32
        self.seed = 42
        self.metric = "accuracy"
        self.data_path = "./data"
        self.train_ratio = 0.8
        self.valid_ratio = 0.1

@pytest.fixture
def dummy_args():
    return DummyArgs()

@pytest.fixture
def monitor_instance(dummy_args):
    return Monitor(dummy_args)

def test_monitor_singleton(dummy_args):
    monitor1 = Monitor(dummy_args)
    monitor2 = Monitor()
    assert monitor1 is monitor2, "Monitor should be a singleton class"

def test_monitor_without_args():
    """Ensure that creating Monitor without args raises an error when no instance exists."""
    Monitor._instance = None  # Reset singleton instance for testing
    error_msg = "Monitor instance cannot be created without args."
    with pytest.raises(ValueError, match=error_msg):
        Monitor()

def test_set_hyperparams(monitor_instance, dummy_args):
    hyperparams = monitor_instance.get_hyperparams()
    assert hyperparams.model == dummy_args.model
    assert hyperparams.embedding_dim == dummy_args.embedding_dim
    assert hyperparams.lr == dummy_args.lr
    assert hyperparams.weight_decay == dummy_args.weight_decay
    assert hyperparams.num_epochs == dummy_args.num_epochs
    assert hyperparams.batch_size == dummy_args.batch_size
    assert hyperparams.seed == dummy_args.seed
    assert hyperparams.metric == dummy_args.metric
    assert hyperparams.data_path == dummy_args.data_path
    assert hyperparams.train_ratio == dummy_args.train_ratio
    assert hyperparams.valid_ratio == dummy_args.valid_ratio