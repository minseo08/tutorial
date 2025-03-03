import pytest
import torch
from tutorial.model import LFM

@pytest.fixture
def model():
    """Fixture to initialize the LFM model."""
    num_users, num_items, embedding_dim = 100, 500, 20
    return LFM(num_users, num_items, embedding_dim)

def test_model_shape(model):
    """Test if model parameters have the correct shapes."""
    num_users, num_items, embedding_dim = 100, 500, 20

    # Validate parameter shapes
    assert model.user_emb.weight.shape == (num_users, embedding_dim)
    assert model.item_emb.weight.shape == (num_items, embedding_dim)
    assert model.user_bias.weight.shape == (num_users, 1)
    assert model.item_bias.weight.shape == (num_items, 1)
    assert model.global_bias.shape == (1,)

def test_model_prediction(model):
    """Test the LFM model with dummy data."""
    num_users, num_items = 100, 500
    batch_size = 10

    # Generate random user and item indices
    users = torch.randint(0, num_users, (batch_size,))
    items = torch.randint(0, num_items, (batch_size,))

    # Forward pass (predict ratings)
    predictions = model(users, items)

    # Validate output shape
    assert predictions.shape == (batch_size,)