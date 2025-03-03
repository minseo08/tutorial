import torch
import torch.nn as nn


class LFM(nn.Module):
    """Latent Factor Model with Global, User, and Item Bias"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 50
    ):
        """Initialize the Latent Factor Model.

        Args:
            num_users (int): Number of unique users.
            num_items (int): Number of unique items.
            embedding_dim (int, optional): Size of embedding vectors. Defaults to 50.
        """
        super(LFM, self).__init__()

        # Global bias term (scalar)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # User and item bias terms (learned scalars per user/item)
        self.user_bias = nn.Embedding(num_users, 1)  # Shape: (num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)  # Shape: (num_items, 1)

        # User and item latent embeddings (learned vectors per user/item)
        # Shape: (num_users, embedding_dim)
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # Initialize parameters
        # Random initialization with small variance
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        # Initialize biases to zero
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor
    ) -> torch.Tensor:
        """Compute predicted ratings.

        Args:
            users (torch.Tensor): Tensor of user indices, shape (batch_size,).
            items (torch.Tensor): Tensor of item indices, shape (batch_size,).

        Returns:
            torch.Tensor: Predicted ratings, shape (batch_size,).
        """
        # Retrieve user and item embeddings
        user_vecs = self.user_emb(users)  # Shape: (batch_size, embedding_dim)
        item_vecs = self.item_emb(items)  # Shape: (batch_size, embedding_dim)

        # Retrieve user and item biases
        user_b = self.user_bias(users).squeeze()  # Shape: (batch_size,)
        item_b = self.item_bias(items).squeeze()  # Shape: (batch_size,)

        # Compute predicted rating
        return self.global_bias + user_b + item_b + (user_vecs * item_vecs).sum(dim=1)