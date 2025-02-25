import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import nn


class Generator(nn.Module):
    """
    A sequence-to-sequence model that generates counterfactual representations.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int, optional): Dimension of the embedding layer. Default is 100.
        hidden_dim (int, optional): Number of hidden units in the LSTM. Default is 256.
        num_layers (int, optional): Number of stacked LSTM layers. Default is 2.
    """

    def __init__(self, vocab_size, device, embedding_dim=100, hidden_dim=128, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim  # Size of hidden state
        self.num_layers = num_layers  # Number of LSTM layers

        # Embedding layer: Converts input tokens into dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer: Processes sequential input data
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        # Linear layer: Outputs a probability distribution over 2 categories (binary classification)
        self.output_layer = nn.Linear(hidden_dim, 2)

        self.device = device

    def forward(self, x):
        """
        Forward pass through the Generator.

        Args:
            x (Tensor): Input sequence tensor (batch_size, sequence_length).

        Returns:
            Tensor: Logits for each token in the sequence (batch_size, sequence_length, 2).
        """
        embedded = self.embeddings(x.to(self.device))  # Convert indices into dense vectors
        out, (hidden, cell) = self.lstm(embedded)  # Pass embeddings through LSTM
        scores = self.output_layer(out)  # Generate class scores for each token
        return scores


class CFDet(nn.Module):
    """
    CFDet: A counterfactual detection model for generating and optimizing counterfactual sequences.

    Attributes:
        exploration_rate (float): Probability of exploration in categorical sampling.
        count_tokens (int): Expected number of token modifications.
        count_pieces (int): Expected number of contiguous modified segments.
        generator (Generator): A neural network module for counterfactual generation.
    """

    def __init__(self, generator, device):
        super(CFDet, self).__init__()
        self.exploration_rate = 0.05  # Exploration rate for stochastic sampling
        self.count_tokens = 3  # Number of modified tokens
        self.count_pieces = 3  # Number of contiguous modified pieces

        # Generator network for counterfactual sequence generation
        self.generator = generator.to(device)

        self.device = device

    def generate(self, x, training=True):
        """
        Generates counterfactual sequences based on input x.

        Args:
            x (Tensor): Input sequence tensor.
            training (bool, optional): If True, uses sampling; otherwise, selects the most probable class.

        Returns:
            Tuple[Tensor, Tensor]: (Generated indices, Negative log probabilities) during training.
            Tuple[Tensor, Tensor]: (Generated indices, Probabilities) during inference.
        """
        # Generate class scores using the generator
        z_scores_ = self.generator(x)
        z_probs_ = F.softmax(z_scores_, dim=-1)  # Convert scores to probabilities

        # Apply exploration rate
        z_prob_ = (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1)

        # Reshape for categorical sampling
        z_prob__ = z_prob_.view(-1, 2)
        sampler = Categorical(z_prob__)

        if training:
            # Sample counterfactuals
            z_ = sampler.sample()  # (num_rows * sequence_length,)
            z = z_.view(z_prob_.size(0), z_prob_.size(1))
            z = z.to(torch.int).to(self.device)

            # Compute negative log probabilities
            neg_log_probs_ = -sampler.log_prob(z_)
            neg_log_probs = neg_log_probs_.view(z_prob_.size(0), z_prob_.size(1))
            return z, neg_log_probs
        else:
            # Select the most probable index
            z__index = torch.max(z_prob__, dim=-1)[1]
            z0 = z__index.view(z_prob_.size(0), z_prob_.size(1))
            z_index = z0.to(torch.int).to(self.device)

            # Get the corresponding probability value
            z__value = torch.max(z_prob__, dim=-1)[0]
            z_value = z__value.to(self.device).float()
            return z_index, z_value
