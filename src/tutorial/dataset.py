from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from loguru import logger
import pandas as pd

class MovieLensDataset(Dataset):
    """PyTorch dataset for MovieLens ratings."""

    def __init__(self, data_path: str):
        """Initializes the dataset by loading the preprocessed data file.

        Args:
            data_path (str): File containing the preprocessed dataset.
        """
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Loads the dataset from a preprocessed Parquet file and converts it 
           to tensors.

        Args:
            data_path (str): File containing the preprocessed dataset.
        """
        logger.info("Loading preprocessed data from {}...", data_path)
        df = pd.read_csv(data_path)

        self.users = torch.tensor(df["userId"].values, dtype=torch.long)
        self.items = torch.tensor(df["movieId"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

        # Store the number of unique users and items
        self.num_users = df["userId"].nunique()
        self.num_items = df["movieId"].nunique()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves a specific sample consisting of user ID, item ID, and rating.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: User ID, item ID, and 
            rating tensors.
        """
        return self.users[idx], self.items[idx], self.ratings[idx]
    
class SplitDataLoader:
    """Handles dataset splitting into training, validation, and test sets."""

    def __init__(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        batch_size: int = 1024,
        seed: int = 42,
    ):
        """Initializes the data loader and splits the dataset.

        Args:
            dataset (Dataset): The dataset to split.
            train_ratio (float): Proportion of the dataset used for training.
            valid_ratio (float): Proportion of the dataset used for validation.
            batch_size (int): Number of samples per batch.
            seed (int): Random seed for reproducibility.
        """
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size
        self.seed = seed

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.train_loader, self.valid_loader, self.test_loader = self._split()

    def _split(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Splits the dataset into training, validation, and test sets and returns 
           DataLoaders.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test 
            data loaders.
        """
        total_size = len(self.dataset)
        train_size = int(self.train_ratio * total_size)
        valid_size = int(self.valid_ratio * total_size)
        test_size = total_size - train_size - valid_size

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, valid_dataset, test_dataset = random_split(
            self.dataset, [train_size, valid_size, test_size], generator
        )

        logger.info(
            "Dataset split: Train: {}, Valid: {}, Test: {}",
            train_size, valid_size, test_size
        )

        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False),
        )
    
