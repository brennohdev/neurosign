"""BiLSTM with dot-product attention for sign-language classification.

Requisitos: 7.1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with dot-product attention classifier.

    Architecture:
        1. BiLSTM encoder  → hidden states (B, T, hidden_size*2)
        2. Dot-product attention → context vector (B, hidden_size*2)
        3. Linear classifier → logits (B, num_classes)

    Args:
        input_size:  Number of input features per timestep (default 84).
        hidden_size: LSTM hidden units per direction (default 128).
        num_layers:  Number of stacked LSTM layers (default 2).
        num_classes: Number of output classes (default 50).
        dropout:     Dropout probability applied between LSTM layers (default 0.3).
    """

    def __init__(
        self,
        input_size: int = 84,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 50,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # 1. BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2. Attention projection: maps each hidden state to a scalar score
        self.attn_linear = nn.Linear(hidden_size * 2, 1)

        # 3. Classifier
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, input_size).

        Returns:
            Logits tensor of shape (B, num_classes).
            No softmax applied — use with ``nn.CrossEntropyLoss``.
        """
        # hidden_states: (B, T, hidden_size*2)
        hidden_states, _ = self.lstm(x)

        # Attention scores: (B, T, 1)
        attn_scores = self.attn_linear(hidden_states)
        attn_weights = F.softmax(attn_scores, dim=1)  # normalise over time axis

        # Context vector: weighted sum over time → (B, hidden_size*2)
        context = (attn_weights * hidden_states).sum(dim=1)

        # Logits: (B, num_classes)
        logits = self.classifier(context)
        return logits
