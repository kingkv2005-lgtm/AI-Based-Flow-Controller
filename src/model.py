import torch.nn as nn


class GRUSoftSensor(nn.Module):
    """
    A single-layer GRU regression model.

    Takes a sequence of process variables (window of timesteps x features)
    and predicts a single scalar output (Flow) from the final hidden state.
    """

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=7,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # take output from the last timestep
        return self.fc(out)
