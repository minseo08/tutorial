import torch
from torch.utils.data import DataLoader
from tutorial.dataset import MovieLensDataset, SplitDataLoader

DATA_PATH = "./data/ML100K.csv"

def test_movielense_dataset():
    """Tests MovieLensDataset initialization and data retrieval with ML100K."""
    dataset = MovieLensDataset(DATA_PATH)
    # Check the data statistics
    assert len(dataset) == 100000
    assert dataset.num_users == 943
    assert dataset.num_items == 1682
    user, item, rating = dataset[0]
    assert isinstance(user, torch.Tensor)
    assert isinstance(item, torch.Tensor)
    assert isinstance(rating, torch.Tensor)

def test_split_dataloader():
    """Tests SplitDataLoader with real data."""
    dataset = MovieLensDataset(DATA_PATH)
    split = SplitDataLoader(
        dataset, train_ratio=0.8, valid_ratio=0.1, batch_size=128, seed=42
    )
    assert isinstance(split.train_loader, DataLoader)
    assert isinstance(split.valid_loader, DataLoader)
    assert isinstance(split.test_loader, DataLoader)

    # Check the size of each loader, where test_loader size
    assert len(split.train_loader.dataset) == 0.8 * len(dataset)
    assert len(split.valid_loader.dataset) == 0.1 * len(dataset)
    assert len(split.test_loader.dataset) == 0.1 * len(dataset)

def test_test_loader_order():
    """Tests that valid_loader and test_loader are fixed when seed is fixed"""
    dataset = MovieLensDataset(DATA_PATH)

    # Note that valid_loader and test_loader should be fixed across different
    # instances of SplitDataLoader when seed is fixed
    split = SplitDataLoader(
        dataset, train_ratio=0.8, valid_ratio=0.1, batch_size=128, seed=42
    )
    second_split = SplitDataLoader(
        dataset, train_ratio=0.8, valid_ratio=0.1, batch_size=128, seed=42
    )

    def check_batch(first_loader, second_loader):
        first_batch = next(iter(first_loader))
        second_batch = next(iter(second_loader))

        # Checking user IDs
        assert torch.allclose(first_batch[0], second_batch[0])
        # Checking item IDs#
        assert torch.allclose(first_batch[1], second_batch[1])
        # Checking ratings
        assert torch.allclose(first_batch[2], second_batch[2])

    check_batch(split.valid_loader, second_split.valid_loader)
    check_batch(split.test_loader, second_split.test_loader)