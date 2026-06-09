"""DouZero-style LSTM action-value models adapted to 4A4.

DouZero's released model uses an LSTM over recent history plus five 512-unit
fully connected layers to score every legal action independently. This module
keeps that shape and replaces DouDizhu-specific feature sizes with the 4A4
feature vectors already produced by this project.
"""
try:
    import torch
    import torch.nn as nn
except ImportError:  # Keep imports cheap on backend machines without torch.
    torch = None
    nn = None

from rl.model import STATE_DIM, ACTION_DIM, HISTORY_DIM


if nn is not None:
    class DouZero4A4LstmModel(nn.Module):
        """One position-specific 4A4 action-value model."""

        def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                     history_dim=HISTORY_DIM, lstm_hidden_dim=128,
                     hidden_dim=512):
            super().__init__()
            self.lstm = nn.LSTM(
                history_dim, lstm_hidden_dim, batch_first=True)
            self.dense1 = nn.Linear(
                state_dim + action_dim + lstm_hidden_dim, hidden_dim)
            self.dense2 = nn.Linear(hidden_dim, hidden_dim)
            self.dense3 = nn.Linear(hidden_dim, hidden_dim)
            self.dense4 = nn.Linear(hidden_dim, hidden_dim)
            self.dense5 = nn.Linear(hidden_dim, hidden_dim)
            self.dense6 = nn.Linear(hidden_dim, 1)

        def _normalize_inputs(self, state_vec, action_vecs, history_vecs):
            if state_vec.dim() == 1:
                state_vec = state_vec.unsqueeze(0)
            if action_vecs.dim() == 2:
                action_vecs = action_vecs.unsqueeze(0)
            if history_vecs.dim() == 2:
                history_vecs = history_vecs.unsqueeze(0)
            return state_vec, action_vecs, history_vecs

        def forward(self, state_vec, action_vecs, history_vecs,
                    return_value=True, flags=None):
            state_vec, action_vecs, history_vecs = self._normalize_inputs(
                state_vec, action_vecs, history_vecs)
            batch, action_count, _ = action_vecs.shape

            lstm_out, _ = self.lstm(history_vecs)
            hist = lstm_out[:, -1, :]
            hist = hist.unsqueeze(1).expand(batch, action_count, hist.shape[-1])
            state = state_vec.unsqueeze(1).expand(
                batch, action_count, state_vec.shape[-1])
            x = torch.cat([hist, state, action_vecs], dim=-1)
            x = x.reshape(batch * action_count, x.shape[-1])

            x = torch.relu(self.dense1(x))
            x = torch.relu(self.dense2(x))
            x = torch.relu(self.dense3(x))
            x = torch.relu(self.dense4(x))
            x = torch.relu(self.dense5(x))
            values = self.dense6(x).reshape(batch, action_count)

            if return_value:
                if batch == 1:
                    return values.squeeze(0)
                return values

            if flags is not None and getattr(flags, 'exp_epsilon', 0) > 0:
                if torch.rand(()) < flags.exp_epsilon:
                    action = torch.randint(action_count, (1,),
                                           device=values.device)[0]
                else:
                    action = torch.argmax(values, dim=-1)[0]
            else:
                action = torch.argmax(values, dim=-1)[0]
            return {'action': action}


    class DouZero4A4Model(nn.Module):
        """Four position-specific models, mirroring DouZero's role wrapper."""

        def __init__(self):
            super().__init__()
            self.models = nn.ModuleList(
                [DouZero4A4LstmModel() for _ in range(4)])

        def forward_for_seat(self, seat, state_vec, action_vecs,
                             history_vecs):
            return self.models[seat % 4](
                state_vec, action_vecs, history_vecs, return_value=True)

        def forward(self, seat, state_vec, action_vecs, history_vecs):
            return self.forward_for_seat(
                seat, state_vec, action_vecs, history_vecs)
else:
    class DouZero4A4LstmModel:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError('DouZero-4A4 model requires torch')

    class DouZero4A4Model:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError('DouZero-4A4 model requires torch')

