"""PyTorch models for DNA sequence classification."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """Convolutional block with batch norm, GELU and max pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 10,
        pool_size: int = 2,
        dropout_rate: float = 0.0,
    ):
        """Initialize ConvBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            pool_size: Size of max pooling window
            dropout_rate: Dropout rate after pooling
        """
        super().__init__()
        padding = kernel_size // 2  # 'SAME' padding equivalent
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class MLPBlock(nn.Module):
    """Dense + GELU + dropout block."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.0):
        """Initialize MLPBlock.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.linear(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        mlp_dim: int = 64,
        dropout_rate: float = 0.2,
    ):
        """Initialize TransformerBlock.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            mlp_dim: Hidden dimension of feedforward network
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.ln1(x)

        # MLP with residual
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)

        return x


class ConvModel(nn.Module):
    """Basic CNN model for binary sequence classification."""

    def __init__(
        self,
        conv_filters: int = 64,
        kernel_size: int = 10,
        dense_units: int = 128,
        dropout_rate: float = 0.0,
    ):
        """Initialize ConvModel.

        Args:
            conv_filters: Number of filters in convolutional layers
            kernel_size: Size of convolutional kernels
            dense_units: Number of units in first dense layer
            dropout_rate: Dropout rate
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(4, conv_filters, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.relu1 = nn.GELU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(
            conv_filters, conv_filters, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(conv_filters)
        self.relu2 = nn.GELU()
        self.pool2 = nn.MaxPool1d(2)

        # After 2 max poolings of size 2: 200 -> 100 -> 50
        # So flattened size = 50 * conv_filters
        flattened_size = 50 * conv_filters

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, dense_units)
        self.relu3 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.relu4 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Output layer
        self.fc_out = nn.Linear(dense_units // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 4)

        Returns:
            Logits of shape (batch_size, 1)
        """
        # Rearrange for Conv1d: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        if self.dropout1 is not None:
            x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu4(x)
        if self.dropout2 is not None:
            x = self.dropout2(x)

        # Output
        logits = self.fc_out(x)

        return logits


class ConvTransformerModel(nn.Module):
    """Model combining CNN, transformer, and MLP blocks."""

    def __init__(
        self,
        num_conv_blocks: int = 2,
        conv_filters: int = 64,
        kernel_size: int = 10,
        num_mlp_blocks: int = 2,
        dense_units: int = 128,
        dropout_rate: float = 0.2,
        num_transformer_blocks: int = 0,
        num_transformer_heads: int = 8,
        transformer_dim: Optional[int] = None,
    ):
        """Initialize ConvTransformerModel.

        Args:
            num_conv_blocks: Number of convolutional blocks
            conv_filters: Number of filters in convolutional layers
            kernel_size: Size of convolutional kernels
            num_mlp_blocks: Number of MLP blocks after flattening
            dense_units: Number of units in first MLP layer
            dropout_rate: Dropout rate
            num_transformer_blocks: Number of transformer blocks (0 for none)
            num_transformer_heads: Number of attention heads
            transformer_dim: Dimension for transformer (default: conv_filters)
        """
        super().__init__()

        self.num_conv_blocks = num_conv_blocks
        self.num_transformer_blocks = num_transformer_blocks
        self.num_mlp_blocks = num_mlp_blocks
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        if transformer_dim is None:
            transformer_dim = conv_filters

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 4
        for i in range(num_conv_blocks):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels,
                    conv_filters,
                    kernel_size,
                    pool_size=2,
                    dropout_rate=dropout_rate,
                )
            )
            in_channels = conv_filters

        # Calculate sequence length after conv blocks
        # Each conv block has MaxPool1d(2), so we divide by 2^num_conv_blocks
        self.seq_len_after_conv = 200 // (2 ** num_conv_blocks)

        # Transformer blocks (if any)
        self.transformer_blocks = nn.ModuleList()
        if num_transformer_blocks > 0:
            # Project conv features to transformer dimension if needed
            if conv_filters != transformer_dim:
                self.conv_to_transformer = nn.Linear(conv_filters, transformer_dim)
            else:
                self.conv_to_transformer = None

            for _ in range(num_transformer_blocks):
                self.transformer_blocks.append(
                    TransformerBlock(
                        transformer_dim,
                        num_heads=num_transformer_heads,
                        mlp_dim=transformer_dim * 2,
                        dropout_rate=dropout_rate,
                    )
                )

            # Flattened size after transformer
            flattened_size = self.seq_len_after_conv * transformer_dim
        else:
            flattened_size = self.seq_len_after_conv * conv_filters

        # MLP blocks
        self.mlp_blocks = nn.ModuleList()
        current_features = flattened_size
        for i in range(num_mlp_blocks):
            next_features = dense_units // (i + 1)
            self.mlp_blocks.append(
                MLPBlock(current_features, next_features, dropout_rate)
            )
            current_features = next_features

        # Output layer
        self.output = nn.Linear(current_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 4)

        Returns:
            Logits of shape (batch_size, 1)
        """
        # Rearrange for Conv1d: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Apply convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Rearrange for transformer: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # Apply transformer blocks if any
        if self.num_transformer_blocks > 0:
            if self.conv_to_transformer is not None:
                # Project each position through the linear layer
                batch_size, seq_len, channels = x.shape
                x = x.reshape(-1, channels)
                x = self.conv_to_transformer(x)
                x = x.reshape(batch_size, seq_len, -1)

            for transformer_block in self.transformer_blocks:
                x = transformer_block(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Apply MLP blocks
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        # Output
        logits = self.output(x)

        return logits


class ConvModelV2(nn.Module):
    """Enhanced CNN with advanced regularization techniques."""

    def __init__(
        self,
        conv_filters: int = 64,
        kernel_size: int = 10,
        dense_units: int = 128,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        """Initialize ConvModelV2.

        Args:
            conv_filters: Number of filters in convolutional layers
            kernel_size: Size of convolutional kernels
            dense_units: Number of units in first dense layer
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        # Convolutional layers with optional normalization
        self.conv1 = nn.Conv1d(4, conv_filters, kernel_size, padding=kernel_size // 2)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(conv_filters)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(conv_filters)

        self.relu1 = nn.GELU()
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1_conv = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.conv2 = nn.Conv1d(
            conv_filters, conv_filters, kernel_size, padding=kernel_size // 2
        )
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(conv_filters)
        if use_layer_norm:
            self.ln2 = nn.LayerNorm(conv_filters)

        self.relu2 = nn.GELU()
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2_conv = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        flattened_size = 50 * conv_filters

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, dense_units)
        self.relu3 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.relu4 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.fc_out = nn.Linear(dense_units // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 4)

        Returns:
            Logits of shape (batch_size, 1)
        """
        x = x.transpose(1, 2)

        # First conv block
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
            x = self.ln1(x)
            x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.pool1(x)
        if self.dropout1_conv is not None:
            x = self.dropout1_conv(x)

        # Second conv block
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
            x = self.ln2(x)
            x = x.transpose(1, 2)
        x = self.relu2(x)
        x = self.pool2(x)
        if self.dropout2_conv is not None:
            x = self.dropout2_conv(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # FC layers
        x = self.fc1(x)
        x = self.relu3(x)
        if self.dropout1 is not None:
            x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu4(x)
        if self.dropout2 is not None:
            x = self.dropout2(x)

        logits = self.fc_out(x)

        return logits
