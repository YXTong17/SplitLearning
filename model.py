import torch.nn as nn


class ClientModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ClientModel, self).__init__()
        self.part1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.part1(x)
        return out


class ServerModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(ServerModel, self).__init__()
        self.part2 = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out = self.part2(x)
        return out
