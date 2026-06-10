"""Transformer action-value model wrapper for DouZero-style 4A4 training.

The old version used four seat-specific LSTM networks. 4A4 is symmetric when
observations are encoded from the acting player's perspective, so this wrapper
now shares one RoPE Transformer model across all seats. The wrapper name stays
for backwards imports, while training no longer needs a separate teacher model.
"""
try:
    import torch.nn as nn
except ImportError:  # Keep imports cheap on backend machines without torch.
    nn = None

from rl.model import CardPolicyNetwork


if nn is not None:
    class DouZero4A4Model(nn.Module):
        """Shared Transformer Q model with the old DouZero wrapper API."""

        def __init__(self):
            super().__init__()
            self.model = CardPolicyNetwork()

        def forward_for_seat(self, seat, state_vec, action_vecs,
                             history_vecs):
            return self.model.forward_for_seat(
                seat, state_vec, action_vecs, history_vecs)

        def forward(self, seat, state_vec, action_vecs, history_vecs):
            return self.forward_for_seat(
                seat, state_vec, action_vecs, history_vecs)
else:
    class DouZero4A4Model:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError('DouZero-4A4 model requires torch')
