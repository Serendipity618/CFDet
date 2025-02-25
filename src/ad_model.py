import torch
import torch.nn as nn


class SVDD(nn.Module):
    """
    Deep SVDD for sequence anomaly detection.

    Args:
        vocab_size (int): The size of the logkeys for embedding.
        embedding_dim (int): The dimension of word embeddings. Default is 8.
        hidden_dim (int): The number of hidden units in the LSTM. Default is 64.
        num_layers (int): The number of stacked LSTM layers. Default is 1.
    """

    def __init__(self, vocab_size, device, embedding_dim=8, hidden_dim=64, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim  # Size of the hidden state
        self.num_layers = num_layers  # Number of LSTM layers

        # Embedding layer to convert input indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer for sequential processing
        self.lstm = nn.LSTM(input_size=embedding_dim,  # Input size = embedding dimension
                            hidden_size=hidden_dim,  # Number of hidden units
                            num_layers=num_layers,  # Number of stacked LSTM layers
                            batch_first=True,  # Input shape (batch, seq_len, feature_dim)
                            bias=False)  # Disable bias for simplicity

        self.device = device

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length)

        Returns:
            Tensor: Output tensor with the mean of LSTM outputs, shape (batch_size,)
        """
        # Initialize hidden state and cell state with random values (on GPU if available)
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_dim).to(self.device)

        # Convert input indices into dense vectors using embedding layer
        embedded = self.embeddings(x.to(self.device))

        # Pass the embeddings through the LSTM layer
        out, (hidden, cell) = self.lstm(embedded, (h0, c0))

        # Compute the mean of LSTM outputs along the sequence dimension and return it
        return torch.squeeze(torch.mean(out, dim=1))
